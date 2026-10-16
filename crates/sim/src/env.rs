//! One reinforcement-learning environment: a vehicle, a task and a seed.

use std::sync::Arc;

use dronelab_core::math::{Quat, Vec3, cos, tanh};
use dronelab_core::{Control, G, MAX_ROTORS, PidControl, PidGains, RateControl, Target};

use crate::airframe::Airframe;
use crate::battery::Battery;
use crate::dynamics::{Drone, Realism, RigidState};
use crate::randomization::Randomization;
use crate::rng::Rng;
use crate::sensors::{Imu, ImuSpec};
use crate::track::{Crossing, LapTimer, Track};

/// How the policy's actions, all in [-1, 1], drive the vehicle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionMode {
    /// One value per rotor: 0 is hover thrust, -1 is off, +1 is twice hover.
    RotorThrust,
    /// Collective thrust (0 = hover) and roll, pitch, yaw rates scaled by the
    /// airframe's `max_rate`. The shared `RateControl` closes the rate loop
    /// at the physics rate, as a flight controller would.
    Ctbr,
    /// Velocity in x, y, z scaled by the vehicle's speed limits. The shared
    /// `PidControl` flies it at the physics rate.
    Velocity,
}

impl ActionMode {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "rotor_thrust" | "thrust" => Some(ActionMode::RotorThrust),
            "ctbr" => Some(ActionMode::Ctbr),
            "velocity" | "vel" => Some(ActionMode::Velocity),
            _ => None,
        }
    }
}

/// Fly to a random point and hold it.
#[derive(Debug, Clone, PartialEq)]
pub struct ReachTask {
    pub target_lo: Vec3,
    pub target_hi: Vec3,
    pub start_lo: Vec3,
    pub start_hi: Vec3,
    /// Largest initial roll/pitch, rad.
    pub start_tilt: f32,
}

/// Fly laps through the gates of a track.
#[derive(Debug, Clone, PartialEq)]
pub struct RaceTask {
    pub track: Track,
    /// Start distance behind gate 0, m.
    pub start_back: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Task {
    Reach(ReachTask),
    Race(RaceTask),
}

/// Everything that defines an environment; shared by all copies in a batch.
#[derive(Debug, Clone)]
pub struct EnvConfig {
    pub airframe: Airframe,
    pub physics_hz: u32,
    /// Policy steps per second; must divide `physics_hz`.
    pub control_hz: u32,
    pub episode_seconds: f32,
    pub action: ActionMode,
    pub realism: Realism,
    pub randomization: Randomization,
    pub imu: ImuSpec,
    pub task: Task,
    /// Position and velocity measurement noise (GNSS/VIO), m and m/s.
    /// Applied when `realism.imu_noise` is on.
    pub pos_noise: f32,
    pub vel_noise: f32,
    /// Observation scales, m and m/s, so observations are about unit size
    /// whether the vehicle is a Crazyflie or an eVTOL.
    pub pos_scale: f32,
    pub vel_scale: f32,
}

impl EnvConfig {
    /// Defaults scaled to the airframe: a reach task, rotor thrust actions,
    /// full realism and moderate randomization.
    pub fn reach(airframe: Airframe) -> Self {
        let r = airframe.radius();
        let h = (5.0 * r).max(1.0);
        let task = Task::Reach(ReachTask {
            target_lo: Vec3::new(-10.0 * r, -10.0 * r, h - 2.0 * r),
            target_hi: Vec3::new(10.0 * r, 10.0 * r, h + 2.0 * r),
            start_lo: Vec3::new(-2.0 * r, -2.0 * r, h - r),
            start_hi: Vec3::new(2.0 * r, 2.0 * r, h + r),
            start_tilt: 0.15,
        });
        Self::with_task(airframe, task)
    }

    /// Defaults for racing on the airframe's default circuit.
    pub fn race(airframe: Airframe) -> Self {
        let track = Track::for_airframe(&airframe);
        let start_back = 15.0 * airframe.radius();
        let mut cfg = Self::with_task(airframe, Task::Race(RaceTask { track, start_back }));
        cfg.episode_seconds = 30.0;
        cfg
    }

    fn with_task(airframe: Airframe, task: Task) -> Self {
        let r = airframe.radius();
        EnvConfig {
            physics_hz: 240,
            control_hz: 30,
            episode_seconds: 10.0,
            action: ActionMode::RotorThrust,
            realism: Realism::FULL,
            randomization: Randomization::default_for(&airframe),
            imu: ImuSpec::default(),
            task,
            pos_noise: 0.01 * r,
            vel_noise: 0.01 * airframe.vehicle.max_speed_xy,
            pos_scale: 10.0 * r,
            vel_scale: airframe.vehicle.max_speed_xy,
            airframe,
        }
    }

    pub fn act_dim(&self) -> usize {
        match self.action {
            ActionMode::RotorThrust => self.airframe.vehicle.rotor_count(),
            ActionMode::Ctbr => 4,
            ActionMode::Velocity => 3,
        }
    }

    pub fn obs_dim(&self) -> usize {
        let base = 18 + self.airframe.vehicle.rotor_count();
        match self.task {
            Task::Reach(_) => base,
            Task::Race(_) => base + 6,
        }
    }

    pub fn substeps(&self) -> u32 {
        (self.physics_hz / self.control_hz).max(1)
    }

    pub fn physics_dt(&self) -> f32 {
        1.0 / self.physics_hz as f32
    }

    pub fn episode_steps(&self) -> u32 {
        (self.episode_seconds * self.control_hz as f32).round() as u32
    }

    /// Checks the settings that would otherwise fail later or silently.
    pub fn validate(&self) -> Result<(), String> {
        if self.control_hz == 0 || !self.physics_hz.is_multiple_of(self.control_hz) {
            return Err(format!(
                "control_hz ({}) must divide physics_hz ({})",
                self.control_hz, self.physics_hz
            ));
        }
        if dronelab_core::Mixer::new(&self.airframe.vehicle).is_none() {
            return Err(format!(
                "rotor layout of {} cannot be mixed",
                self.airframe.vehicle.name
            ));
        }
        if let Task::Race(r) = &self.task
            && r.track.gates.len() < 2
        {
            return Err("a race track needs at least 2 gates".into());
        }
        Ok(())
    }
}

/// Outcome of one policy step.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct StepInfo {
    pub reward: f32,
    pub terminated: bool,
    pub truncated: bool,
    pub crashed: bool,
    pub gate_passed: bool,
    /// Set when this step completed a lap, s.
    pub lap_time: Option<f64>,
}

impl StepInfo {
    pub fn done(&self) -> bool {
        self.terminated || self.truncated
    }
}

#[derive(Debug, Clone)]
pub struct Env {
    cfg: Arc<EnvConfig>,
    pub drone: Drone,
    rng: Rng,
    /// Seed of the current episode's first reset, for replays.
    pub seed: u64,
    pub target: Vec3,
    yaw_hold: f32,
    pid: PidControl,
    expert: PidControl,
    rate: RateControl,
    pub lap: LapTimer,
    step_count: u32,
    prev_action: [f32; MAX_ROTORS],
    prev_dist: f32,
}

impl Env {
    /// Panics if the config does not [`EnvConfig::validate`].
    pub fn new(cfg: Arc<EnvConfig>, seed: u64) -> Self {
        cfg.validate().expect("invalid EnvConfig");
        let v = cfg.airframe.vehicle;
        let pid = PidControl::new(v, G).expect("validated");
        // The expert acts only at the policy rate, so keep its attitude
        // loop slow enough for that sample time.
        let att_wn = (1.0 / (3.0 * v.motor_tau)).min(0.3 * cfg.control_hz as f32);
        let expert = PidControl::with_gains(v, G, PidGains::from_bandwidth(att_wn, att_wn / 5.0))
            .expect("validated");
        let rate = RateControl::new(v).expect("validated");
        let drone = Drone::new(
            &cfg.airframe,
            crate::dynamics::Physical::nominal(&cfg.airframe),
            RigidState::default(),
            Battery::new(cfg.airframe.battery, 1.0),
            crate::sensors::Wind::CALM,
            Imu::ideal(),
        );
        let mut env = Env {
            cfg,
            drone,
            rng: Rng::new(seed),
            seed,
            target: Vec3::ZERO,
            yaw_hold: 0.0,
            pid,
            expert,
            rate,
            lap: LapTimer::default(),
            step_count: 0,
            prev_action: [0.0; MAX_ROTORS],
            prev_dist: 0.0,
        };
        env.reset(Some(seed));
        env
    }

    pub fn config(&self) -> &EnvConfig {
        &self.cfg
    }

    /// Starts a new episode. With a seed the episode is fully determined by
    /// it; without, the environment's random stream continues.
    pub fn reset(&mut self, seed: Option<u64>) {
        if let Some(s) = seed {
            self.rng = Rng::new(s);
            self.seed = s;
        }
        let cfg = Arc::clone(&self.cfg);
        let af = &cfg.airframe;
        let rng = &mut self.rng;
        let (phys, wind, soc) = cfg.randomization.sample(af, rng);
        let imu = if cfg.realism.imu_noise {
            Imu::new(cfg.imu, cfg.randomization.imu_bias_scale, rng)
        } else {
            Imu::ideal()
        };

        let pick = |rng: &mut Rng, lo: Vec3, hi: Vec3| {
            Vec3::new(
                rng.range(lo.x, hi.x),
                rng.range(lo.y, hi.y),
                rng.range(lo.z, hi.z),
            )
        };
        let (pos, yaw, tilt) = match &cfg.task {
            Task::Reach(t) => {
                self.target = pick(rng, t.target_lo, t.target_hi);
                let yaw = rng.range(-core::f32::consts::PI, core::f32::consts::PI);
                (pick(rng, t.start_lo, t.start_hi), yaw, t.start_tilt)
            }
            Task::Race(t) => {
                let g = &t.track.gates[0];
                self.target = g.center();
                (g.center() - g.normal() * t.start_back, g.yaw, 0.05)
            }
        };
        let quat = Quat::from_rpy(rng.range(-tilt, tilt), rng.range(-tilt, tilt), yaw);
        let ang_vel = Vec3::new(
            rng.range(-0.1, 0.1),
            rng.range(-0.1, 0.1),
            rng.range(-0.1, 0.1),
        );
        let state = RigidState {
            pos,
            vel: Vec3::ZERO,
            quat,
            ang_vel,
        };
        self.drone = Drone::new(af, phys, state, Battery::new(af.battery, soc), wind, imu);
        self.drone.imu.last.gyro = ang_vel;
        self.yaw_hold = yaw;
        self.pid.reset();
        self.expert.reset();
        self.lap = LapTimer::default();
        self.step_count = 0;
        self.prev_action = [0.0; MAX_ROTORS];
        self.prev_dist = (self.target - pos).norm();
    }

    /// Simulated time in this episode, s.
    pub fn time(&self) -> f64 {
        self.drone.steps as f64 / self.cfg.physics_hz as f64
    }

    /// Writes the observation (see the crate docs for the layout).
    pub fn observe(&mut self, out: &mut [f32]) {
        let cfg = &self.cfg;
        let af = &cfg.airframe;
        let d = &self.drone;
        let mut pos = d.state.pos;
        let mut vel = d.state.vel;
        if cfg.realism.imu_noise {
            pos +=
                Vec3::new(self.rng.normal(), self.rng.normal(), self.rng.normal()) * cfg.pos_noise;
            vel +=
                Vec3::new(self.rng.normal(), self.rng.normal(), self.rng.normal()) * cfg.vel_noise;
        }
        let rel = (self.target - pos) * (1.0 / cfg.pos_scale);
        let v = vel * (1.0 / cfg.vel_scale);
        let gyro = d.imu.last.gyro * (1.0 / af.max_rate);
        out[0..3].copy_from_slice(&rel.to_array());
        out[3..6].copy_from_slice(&v.to_array());
        out[6..15].copy_from_slice(&d.state.quat.to_mat().to_array());
        out[15..18].copy_from_slice(&gyro.to_array());
        let n = af.vehicle.rotor_count();
        for i in 0..n {
            out[18 + i] = d.omega[i] / af.omega_max;
        }
        if let Task::Race(t) = &cfg.task {
            let gates = &t.track.gates;
            let next = &gates[self.lap.next_gate];
            let after = gates[(self.lap.next_gate + 1) % gates.len()].center();
            out[18 + n..21 + n].copy_from_slice(&next.normal().to_array());
            out[21 + n..24 + n]
                .copy_from_slice(&((after - pos) * (1.0 / cfg.pos_scale)).to_array());
        }
    }

    /// Rotor commands (fraction of max thrust) for one physics step.
    fn rotor_commands(&mut self, action: &[f32], cmd: &mut [f32; MAX_ROTORS]) {
        let cfg = &self.cfg;
        let af = &cfg.airframe;
        let v = &af.vehicle;
        let n = v.rotor_count();
        let dt = cfg.physics_dt();
        let a = |i: usize| action[i].clamp(-1.0, 1.0);
        let thrust = match cfg.action {
            ActionMode::RotorThrust => {
                let h = af.hover_fraction();
                for (i, c) in cmd[..n].iter_mut().enumerate() {
                    *c = (h * (1.0 + a(i))).clamp(0.0, 1.0);
                }
                return;
            }
            ActionMode::Ctbr => {
                let collective = (v.weight() * (1.0 + a(0))).clamp(0.0, v.max_total_thrust());
                let rates = Vec3::new(a(1), a(2), a(3)) * af.max_rate;
                self.rate
                    .compute(collective, rates, self.drone.imu.last.gyro)
                    .thrust
            }
            ActionMode::Velocity => {
                let vel = Vec3::new(
                    a(0) * v.max_speed_xy,
                    a(1) * v.max_speed_xy,
                    a(2) * v.max_speed_z,
                );
                // Position target = current position: only the velocity loop
                // (PI on velocity error) acts.
                let pos = self.drone.state.pos;
                let target = Target {
                    pos,
                    vel,
                    rpy: Vec3::new(0.0, 0.0, self.yaw_hold),
                    ..Default::default()
                };
                self.pid
                    .compute_control(dt, &self.drone.measured_state(), &target)
                    .thrust
            }
        };
        for (i, r) in v.rotors().iter().enumerate() {
            cmd[i] = (thrust[i] / r.max_thrust).clamp(0.0, 1.0);
        }
    }

    /// Applies `action` for one control period and writes the next observation.
    pub fn step(&mut self, action: &[f32], obs: &mut [f32]) -> StepInfo {
        let cfg = Arc::clone(&self.cfg);
        let af = &cfg.airframe;
        let r = af.radius();
        let mut info = StepInfo::default();
        let mut cmd = [0.0f32; MAX_ROTORS];
        let mut frame_hit = false;

        for _ in 0..cfg.substeps() {
            self.rotor_commands(action, &mut cmd);
            let prev = self.drone.state.pos;
            self.drone
                .step(af, &cfg.realism, &cmd, cfg.physics_dt(), &mut self.rng);
            if let Task::Race(t) = &cfg.task {
                let gates = &t.track.gates;
                match gates[self.lap.next_gate].crossing(prev, self.drone.state.pos, r) {
                    Crossing::Through => {
                        info.gate_passed = true;
                        info.lap_time = info.lap_time.or(self.lap.pass(gates.len(), self.time()));
                        self.target = gates[self.lap.next_gate].center();
                        self.prev_dist = (self.target - self.drone.state.pos).norm();
                        info.reward += if info.lap_time.is_some() { 4.0 } else { 1.0 };
                    }
                    Crossing::Frame => frame_hit = true,
                    Crossing::None | Crossing::Missed => {}
                }
            }
            if self.drone.crashed || frame_hit {
                break;
            }
        }
        self.step_count += 1;

        let d = &self.drone;
        let tilt_limit = (2.5 * af.vehicle.max_tilt).min(1.5);
        let dist = (self.target - d.state.pos).norm();
        let crashed = d.crashed || frame_hit || d.up_dot() < cos(tilt_limit);
        let lost = dist > 4.0 * cfg.pos_scale + 2.0 * self.track_radius();
        let rate_cost = 0.05 * (d.imu.last.gyro.norm() / af.max_rate).min(4.0);
        let n = cfg.act_dim();
        let smooth_cost = 0.02
            * (0..n)
                .map(|i| (action[i] - self.prev_action[i]).powi(2))
                .sum::<f32>()
            / n as f32;
        info.reward += match cfg.task {
            // Closeness plus progress towards the target: the progress term
            // gives an immediate signal for moving the right way, which
            // matters for slow vehicles that take seconds to close in.
            Task::Reach(_) => 1.0 - tanh(dist / cfg.pos_scale) + 10.0 * (self.prev_dist - dist) / cfg.pos_scale,
            Task::Race(_) => (self.prev_dist - dist) / cfg.pos_scale,
        } - rate_cost
            - smooth_cost;
        self.prev_dist = dist;
        self.prev_action[..n].copy_from_slice(&action[..n]);

        if crashed || lost {
            info.crashed = crashed;
            info.terminated = true;
            info.reward = -10.0;
        } else if self.step_count >= cfg.episode_steps() {
            info.truncated = true;
        }
        self.observe(obs);
        info
    }

    fn track_radius(&self) -> f32 {
        match &self.cfg.task {
            Task::Race(t) => t
                .track
                .gates
                .iter()
                .map(|g| Vec3::new(g.center[0], g.center[1], 0.0).norm())
                .fold(0.0, f32::max),
            Task::Reach(_) => 0.0,
        }
    }

    /// What the built-in PID controller would do, in this env's action
    /// space (`None` for CTBR). Useful as a baseline and for imitation.
    pub fn expert_action(&mut self, out: &mut [f32]) -> Option<()> {
        let cfg = Arc::clone(&self.cfg);
        let af = &cfg.airframe;
        let v = &af.vehicle;
        let pos = self.drone.state.pos;
        let mut goal = self.target;
        // Velocity feed-forward: none when holding a point, cruise speed
        // towards the pursuit point when racing.
        let mut feed = Vec3::ZERO;
        if let Task::Race(t) = &cfg.task {
            // Pure pursuit along the gate's centre line: aim ahead of our
            // projection onto it, so the vehicle lines up first and then
            // flies through the middle without stopping.
            let g = &t.track.gates[self.lap.next_gate];
            let (c, n) = (g.center(), g.normal());
            // Cruise at what the track's turns allow at half the tilt
            // limit; look ahead about one position-loop time constant.
            let turn = (0.5 * G * libm::tanf(v.max_tilt) * self.track_radius()).sqrt();
            let cruise = (0.4 * v.max_speed_xy).min(0.6 * turn);
            let pos_wn = self.expert.gains.d_pos.x / 1.8;
            let r = af.radius();
            let lead = (cruise / pos_wn).clamp(4.0 * r, 8.0 * r);
            let along = (pos - c).dot(n);
            // Past the plane without getting through: come round again.
            let aim = if along > 0.0 {
                -lead
            } else {
                (along + lead).min(lead)
            };
            goal = c + n * aim;
            feed = (goal - pos).normalized_or(n) * cruise;
            feed.z = feed.z.clamp(-v.max_speed_z, v.max_speed_z);
        }
        match cfg.action {
            ActionMode::RotorThrust => {
                let target = Target {
                    pos: goal,
                    vel: feed,
                    rpy: Vec3::new(0.0, 0.0, self.yaw_hold),
                    ..Default::default()
                };
                let o = self.expert.compute_control(
                    1.0 / cfg.control_hz as f32,
                    &self.drone.measured_state(),
                    &target,
                );
                let h = af.hover_fraction();
                for (i, r) in v.rotors().iter().enumerate() {
                    out[i] = (o.thrust[i] / r.max_thrust / h - 1.0).clamp(-1.0, 1.0);
                }
            }
            ActionMode::Velocity => {
                let g = self.expert.gains;
                let e = goal - pos;
                let vel = feed
                    + Vec3::new(
                        g.p_pos.x / g.d_pos.x * e.x,
                        g.p_pos.y / g.d_pos.y * e.y,
                        g.p_pos.z / g.d_pos.z * e.z,
                    );
                out[0] = (vel.x / v.max_speed_xy).clamp(-1.0, 1.0);
                out[1] = (vel.y / v.max_speed_xy).clamp(-1.0, 1.0);
                out[2] = (vel.z / v.max_speed_z).clamp(-1.0, 1.0);
            }
            ActionMode::Ctbr => return None,
        }
        Some(())
    }
}

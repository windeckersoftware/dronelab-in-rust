//! `BaseAviary` and its subclasses: `CtrlAviary`, `VelocityAviary`,
//! `BaseRLAviary`, `HoverAviary` and `MultiHoverAviary`.
//!
//! Upstream behaviour that the port keeps on purpose:
//! - `reset()` does not reset the PID controllers or the RL action buffer;
//! - RL action types that use a PID always build a CF2X controller;
//! - in `DYN` mode drones are not stopped by the ground;
//! - an episode is truncated when `step_counter / pyb_freq > 8` before the
//!   counter advances, so it lasts 242 steps at 240/30 Hz;
//! - RL observations hold float32-rounded kinematics next to float64
//!   actions, as NumPy's `hstack` produces.

use std::collections::VecDeque;

use dronelab::DroneModel;

use crate::control::{ControlError, DslPidControl, Target};
use crate::math::*;
use crate::params::{G, Params};
use crate::physics::{self, Applied, Body, Physics};

/// `ActionType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionType {
    Rpm,
    Pid,
    Vel,
    OneDRpm,
    OneDPid,
}

impl ActionType {
    pub fn value(self) -> &'static str {
        match self {
            ActionType::Rpm => "rpm",
            ActionType::Pid => "pid",
            ActionType::Vel => "vel",
            ActionType::OneDRpm => "one_d_rpm",
            ActionType::OneDPid => "one_d_pid",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        [
            ActionType::Rpm,
            ActionType::Pid,
            ActionType::Vel,
            ActionType::OneDRpm,
            ActionType::OneDPid,
        ]
        .into_iter()
        .find(|a| a.value() == s)
    }

    /// Action size per drone.
    pub fn size(self) -> usize {
        match self {
            ActionType::Rpm | ActionType::Vel => 4,
            ActionType::Pid => 3,
            ActionType::OneDRpm | ActionType::OneDPid => 1,
        }
    }

    fn uses_pid(self) -> bool {
        matches!(
            self,
            ActionType::Pid | ActionType::Vel | ActionType::OneDPid
        )
    }
}

/// Which upstream environment class this is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnvKind {
    /// `CtrlAviary`: RPM in, 20-value state out.
    Ctrl,
    /// `VelocityAviary`: `[vx, vy, vz, speed fraction]` through DSL PID.
    Velocity,
    /// `HoverAviary` (one drone, target `[0, 0, 1]`).
    Hover(ActionType),
    /// `MultiHoverAviary` (targets 1/(i+1) m above each start).
    MultiHover(ActionType),
}

impl EnvKind {
    fn rl_action(self) -> Option<ActionType> {
        match self {
            EnvKind::Hover(a) | EnvKind::MultiHover(a) => Some(a),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AviaryConfig {
    pub kind: EnvKind,
    pub drone_model: DroneModel,
    pub num_drones: usize,
    pub neighbourhood_radius: f64,
    pub initial_xyzs: Option<Vec<Vec3>>,
    pub initial_rpys: Option<Vec<Vec3>>,
    pub physics: Physics,
    pub pyb_freq: u32,
    pub ctrl_freq: u32,
}

impl AviaryConfig {
    /// The upstream constructor defaults for `kind`.
    pub fn new(kind: EnvKind) -> Self {
        let (num_drones, ctrl_freq) = match kind {
            EnvKind::Ctrl | EnvKind::Velocity => (1, 240),
            EnvKind::Hover(_) => (1, 30),
            EnvKind::MultiHover(_) => (2, 30),
        };
        AviaryConfig {
            kind,
            drone_model: DroneModel::Cf2x,
            num_drones,
            neighbourhood_radius: f64::INFINITY,
            initial_xyzs: None,
            initial_rpys: None,
            physics: Physics::Pyb,
            pyb_freq: 240,
            ctrl_freq,
        }
    }
}

#[derive(Debug, Clone)]
pub enum AviaryError {
    Config(String),
    Control(ControlError),
}

impl std::fmt::Display for AviaryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AviaryError::Config(s) => f.write_str(s),
            AviaryError::Control(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for AviaryError {}

/// What `BaseAviary._updateAndStoreKinematicInformation` keeps per drone.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Kinematics {
    pub pos: Vec3,
    pub quat: QuatXyzw,
    pub rpy: Vec3,
    pub vel: Vec3,
    pub ang_v: Vec3,
}

/// One `step()`'s results. `obs` is row-major `(num_drones, obs_dim)`.
#[derive(Debug, Clone, PartialEq)]
pub struct StepResult {
    pub obs: Vec<f64>,
    pub reward: f64,
    pub terminated: bool,
    pub truncated: bool,
}

#[derive(Debug, Clone)]
pub struct Aviary {
    pub cfg: AviaryConfig,
    pub params: Params,
    pub init_xyzs: Vec<Vec3>,
    pub init_rpys: Vec<Vec3>,
    pub pyb_steps_per_ctrl: u32,
    pub ctrl_timestep: f64,
    pub pyb_timestep: f64,
    pub step_counter: u64,
    pub stored: Vec<Kinematics>,
    pub bodies: Vec<Body>,
    pub last_clipped_action: Vec<[f64; 4]>,
    /// `DYN` only: body rates carried between steps.
    rpy_rates: Vec<Vec3>,
    pub ctrl: Vec<DslPidControl>,
    /// `BaseRLAviary.action_buffer` (each entry `num_drones * size`).
    pub action_buffer: VecDeque<Vec<f64>>,
    pub action_buffer_size: usize,
    pub target_pos: Vec<Vec3>,
    pub episode_len_sec: f64,
    pub speed_limit: f64,
    /// Forces applied in the last step, in order (for tests).
    pub record_forces: bool,
    pub applied: Vec<(usize, Applied)>,
    /// Steps in which a drone touched the ground (approximate contact).
    pub contacts: u64,
}

impl Aviary {
    pub fn new(cfg: AviaryConfig) -> Result<Self, AviaryError> {
        if cfg.ctrl_freq == 0 || !cfg.pyb_freq.is_multiple_of(cfg.ctrl_freq) {
            return Err(AviaryError::Config(
                "[ERROR] in BaseAviary.__init__(), pyb_freq is not divisible by env_freq.".into(),
            ));
        }
        if matches!(cfg.kind, EnvKind::Hover(_)) && cfg.num_drones != 1 {
            return Err(AviaryError::Config(
                "HoverAviary has exactly one drone".into(),
            ));
        }
        let params =
            Params::load(cfg.drone_model).map_err(|e| AviaryError::Config(e.to_string()))?;
        let n = cfg.num_drones;
        let init_xyzs = match &cfg.initial_xyzs {
            None => params.default_init_xyzs(n),
            Some(v) if v.len() == n => v.clone(),
            Some(_) => return Err(AviaryError::Config("invalid initial_xyzs in BaseAviary.__init__(), try initial_xyzs.reshape(NUM_DRONES,3)".into())),
        };
        let init_rpys = match &cfg.initial_rpys {
            None => vec![[0.0; 3]; n],
            Some(v) if v.len() == n => v.clone(),
            Some(_) => return Err(AviaryError::Config("invalid initial_rpys in BaseAviary.__init__(), try initial_rpys.reshape(NUM_DRONES,3)".into())),
        };
        let needs_pid =
            cfg.kind == EnvKind::Velocity || cfg.kind.rl_action().is_some_and(ActionType::uses_pid);
        let ctrl = if needs_pid {
            if !matches!(cfg.drone_model, DroneModel::Cf2x | DroneModel::Cf2p) {
                return Err(AviaryError::Config("in BaseRLAviary.__init__(), no controller is available for the specified drone_model".into()));
            }
            (0..n)
                .map(|_| DslPidControl::new(DroneModel::Cf2x, G).map_err(AviaryError::Control))
                .collect::<Result<_, _>>()?
        } else {
            Vec::new()
        };
        let action_buffer_size = (cfg.ctrl_freq / 2) as usize;
        let mut action_buffer = VecDeque::with_capacity(action_buffer_size);
        if let Some(a) = cfg.kind.rl_action() {
            for _ in 0..action_buffer_size {
                action_buffer.push_back(vec![0.0; n * a.size()]);
            }
        }
        let target_pos = match cfg.kind {
            EnvKind::Hover(_) => vec![[0.0, 0.0, 1.0]],
            EnvKind::MultiHover(_) => init_xyzs
                .iter()
                .enumerate()
                .map(|(i, p)| [p[0], p[1], p[2] + 1.0 / (i as f64 + 1.0)])
                .collect(),
            _ => Vec::new(),
        };
        let speed_limit = 0.03 * params.max_speed_kmh * (1000.0 / 3600.0);
        let mut av = Aviary {
            pyb_steps_per_ctrl: cfg.pyb_freq / cfg.ctrl_freq,
            ctrl_timestep: 1.0 / cfg.ctrl_freq as f64,
            pyb_timestep: 1.0 / cfg.pyb_freq as f64,
            step_counter: 0,
            stored: vec![Kinematics::default(); n],
            bodies: Vec::new(),
            last_clipped_action: vec![[0.0; 4]; n],
            rpy_rates: vec![[0.0; 3]; n],
            ctrl,
            action_buffer,
            action_buffer_size,
            target_pos,
            episode_len_sec: 8.0,
            speed_limit,
            record_forces: false,
            applied: Vec::new(),
            contacts: 0,
            init_xyzs,
            init_rpys,
            params,
            cfg,
        };
        av.housekeeping();
        Ok(av)
    }

    pub fn num_drones(&self) -> usize {
        self.cfg.num_drones
    }

    /// Action size per drone.
    pub fn act_dim(&self) -> usize {
        match self.cfg.kind.rl_action() {
            Some(a) => a.size(),
            None => 4,
        }
    }

    /// Observation size per drone.
    pub fn obs_dim(&self) -> usize {
        match self.cfg.kind.rl_action() {
            Some(a) => 12 + self.action_buffer_size * a.size(),
            None => 20,
        }
    }

    fn housekeeping(&mut self) {
        self.step_counter = 0;
        self.last_clipped_action = vec![[0.0; 4]; self.cfg.num_drones];
        self.rpy_rates = vec![[0.0; 3]; self.cfg.num_drones];
        self.bodies = self
            .init_xyzs
            .iter()
            .zip(&self.init_rpys)
            .map(|(p, r)| Body::new(*p, pb_quat_from_euler(*r)))
            .collect();
        self.update_and_store();
    }

    fn update_and_store(&mut self) {
        for (k, b) in self.stored.iter_mut().zip(&self.bodies) {
            let quat = pb_readback_quat(b.quat);
            *k = Kinematics {
                pos: b.pos,
                quat,
                rpy: pb_euler(quat),
                vel: b.vel,
                ang_v: b.ang_vel,
            };
        }
    }

    /// `_getDroneStateVector`.
    pub fn state_vector(&self, i: usize) -> [f64; 20] {
        let k = &self.stored[i];
        let mut s = [0.0; 20];
        s[0..3].copy_from_slice(&k.pos);
        s[3..7].copy_from_slice(&k.quat);
        s[7..10].copy_from_slice(&k.rpy);
        s[10..13].copy_from_slice(&k.vel);
        s[13..16].copy_from_slice(&k.ang_v);
        s[16..20].copy_from_slice(&self.last_clipped_action[i]);
        s
    }

    /// `reset()`: returns the initial observation.
    pub fn reset(&mut self) -> Vec<f64> {
        self.housekeeping();
        self.compute_obs()
    }

    fn pid(&mut self, k: usize, target: Target) -> [f64; 4] {
        let s = self.state_vector(k);
        let dt = self.ctrl_timestep;
        self.ctrl[k].compute_control_from_state(dt, &s, &target).rpm
    }

    /// `_preprocessAction`: actions (row-major, `num_drones * act_dim`) to RPM.
    fn preprocess_action(&mut self, action: &[f64]) -> Vec<[f64; 4]> {
        let n = self.cfg.num_drones;
        let ad = self.act_dim();
        let mut rpm = vec![[0.0; 4]; n];
        match self.cfg.kind {
            EnvKind::Ctrl => {
                for (k, r) in rpm.iter_mut().enumerate() {
                    for j in 0..4 {
                        r[j] = clip(action[k * 4 + j], 0.0, self.params.max_rpm);
                    }
                }
            }
            EnvKind::Velocity => {
                for (k, r) in rpm.iter_mut().enumerate() {
                    *r = self.velocity_rpm(k, &action[k * 4..k * 4 + 4]);
                }
            }
            EnvKind::Hover(act) | EnvKind::MultiHover(act) => {
                if self.action_buffer.len() == self.action_buffer_size
                    && self.action_buffer_size > 0
                {
                    self.action_buffer.pop_front();
                }
                if self.action_buffer_size > 0 {
                    self.action_buffer.push_back(action.to_vec());
                }
                let hover = self.params.hover_rpm;
                for (k, r) in rpm.iter_mut().enumerate() {
                    let t = &action[k * ad..(k + 1) * ad];
                    *r = match act {
                        ActionType::Rpm => [0, 1, 2, 3].map(|j| hover * (1.0 + 0.05 * t[j])),
                        ActionType::OneDRpm => [hover * (1.0 + 0.05 * t[0]); 4],
                        ActionType::Pid => {
                            let s = self.state_vector(k);
                            let next =
                                calculate_next_step([s[0], s[1], s[2]], [t[0], t[1], t[2]], 1.0);
                            self.pid(
                                k,
                                Target {
                                    pos: next,
                                    ..Default::default()
                                },
                            )
                        }
                        ActionType::Vel => self.velocity_rpm(k, t),
                        ActionType::OneDPid => {
                            let s = self.state_vector(k);
                            self.pid(
                                k,
                                Target {
                                    pos: [s[0], s[1], s[2] + 0.1 * t[0]],
                                    ..Default::default()
                                },
                            )
                        }
                    };
                }
            }
        }
        rpm
    }

    fn velocity_rpm(&mut self, k: usize, t: &[f64]) -> [f64; 4] {
        let s = self.state_vector(k);
        let v = [t[0], t[1], t[2]];
        let n = norm(v);
        let unit = if n != 0.0 {
            scale(v, 1.0 / n)
        } else {
            [0.0; 3]
        };
        let target = Target {
            pos: [s[0], s[1], s[2]],
            rpy: [0.0, 0.0, s[9]],
            vel: scale(unit, self.speed_limit * t[3].abs()),
            ..Default::default()
        };
        self.pid(k, target)
    }

    /// `step()`.
    pub fn step(&mut self, action: &[f64]) -> StepResult {
        assert_eq!(
            action.len(),
            self.cfg.num_drones * self.act_dim(),
            "action size"
        );
        let clipped = self.preprocess_action(action);
        let physics = self.cfg.physics;
        let refresh = self.pyb_steps_per_ctrl > 1 && physics != Physics::Pyb;
        let dt = self.pyb_timestep;
        self.applied.clear();
        let mut forces = Vec::new();
        for _ in 0..self.pyb_steps_per_ctrl {
            if refresh {
                self.update_and_store();
            }
            let positions: Vec<Vec3> = self.stored.iter().map(|k| k.pos).collect();
            for i in 0..self.cfg.num_drones {
                if physics == Physics::Dyn {
                    // `_dynamics` integrates from the stored (read-back) state.
                    let k = self.stored[i];
                    let from = Body {
                        pos: k.pos,
                        quat: k.quat,
                        vel: k.vel,
                        ..self.bodies[i]
                    };
                    let (b, rates) =
                        physics::dyn_step(&self.params, &clipped[i], &from, self.rpy_rates[i], dt);
                    self.bodies[i] = b;
                    self.rpy_rates[i] = rates;
                    continue;
                }
                forces.clear();
                let k = self.stored[i];
                physics::substep_forces(
                    &self.params,
                    physics,
                    &clipped[i],
                    &self.last_clipped_action[i],
                    &self.bodies[i],
                    k.quat,
                    k.rpy,
                    k.vel,
                    &positions,
                    i,
                    &mut forces,
                );
                for a in &forces {
                    physics::apply(&self.params, &mut self.bodies[i], *a);
                }
                if self.record_forces {
                    self.applied.extend(forces.iter().map(|a| (i, *a)));
                }
            }
            if physics != Physics::Dyn {
                for b in &mut self.bodies {
                    physics::pyb_step(&self.params, b, dt);
                    if physics::ground_contact(&self.params, b, dt) {
                        self.contacts += 1;
                    }
                }
            }
            self.last_clipped_action.clone_from(&clipped);
        }
        self.update_and_store();
        let obs = self.compute_obs();
        let reward = self.compute_reward();
        let terminated = self.compute_terminated();
        let truncated = self.compute_truncated();
        self.step_counter += self.pyb_steps_per_ctrl as u64;
        StepResult {
            obs,
            reward,
            terminated,
            truncated,
        }
    }

    /// `_computeObs`.
    pub fn compute_obs(&self) -> Vec<f64> {
        let n = self.cfg.num_drones;
        match self.cfg.kind.rl_action() {
            None => (0..n).flat_map(|i| self.state_vector(i)).collect(),
            Some(a) => {
                let mut out = Vec::with_capacity(n * self.obs_dim());
                for i in 0..n {
                    let s = self.state_vector(i);
                    for idx in (0..3).chain(7..10).chain(10..13).chain(13..16) {
                        out.push(s[idx] as f32 as f64);
                    }
                    for entry in &self.action_buffer {
                        out.extend_from_slice(&entry[i * a.size()..(i + 1) * a.size()]);
                    }
                }
                out
            }
        }
    }

    fn compute_reward(&self) -> f64 {
        match self.cfg.kind {
            EnvKind::Ctrl | EnvKind::Velocity => -1.0,
            EnvKind::Hover(_) | EnvKind::MultiHover(_) => (0..self.cfg.num_drones)
                .map(|i| (2.0 - norm(sub(self.target_pos[i], self.stored[i].pos)).powi(4)).max(0.0))
                .sum(),
        }
    }

    fn compute_terminated(&self) -> bool {
        match self.cfg.kind {
            EnvKind::Ctrl | EnvKind::Velocity => false,
            EnvKind::Hover(_) | EnvKind::MultiHover(_) => {
                let dist: f64 = (0..self.cfg.num_drones)
                    .map(|i| norm(sub(self.target_pos[i], self.stored[i].pos)))
                    .sum();
                dist < 0.0001
            }
        }
    }

    fn compute_truncated(&self) -> bool {
        let bound = match self.cfg.kind {
            EnvKind::Ctrl | EnvKind::Velocity => return false,
            EnvKind::Hover(_) => 1.5,
            EnvKind::MultiHover(_) => 2.0,
        };
        let out = self.stored.iter().any(|k| {
            k.pos[0].abs() > bound
                || k.pos[1].abs() > bound
                || k.pos[2] > 2.0
                || k.rpy[0].abs() > 0.4
                || k.rpy[1].abs() > 0.4
        });
        out || self.step_counter as f64 / self.cfg.pyb_freq as f64 > self.episode_len_sec
    }

    /// `_getAdjacencyMatrix` (row-major).
    pub fn adjacency_matrix(&self) -> Vec<f64> {
        let n = self.cfg.num_drones;
        let mut m = vec![0.0; n * n];
        for i in 0..n {
            m[i * n + i] = 1.0;
            for j in i + 1..n {
                if norm(sub(self.stored[i].pos, self.stored[j].pos)) < self.cfg.neighbourhood_radius
                {
                    m[i * n + j] = 1.0;
                    m[j * n + i] = 1.0;
                }
            }
        }
        m
    }
}

/// `BaseAviary._calculateNextStep`.
pub fn calculate_next_step(current: Vec3, destination: Vec3, step_size: f64) -> Vec3 {
    let direction = sub(destination, current);
    let distance = norm(direction);
    if distance <= step_size {
        return destination;
    }
    add(current, scale(scale(direction, 1.0 / distance), step_size))
}

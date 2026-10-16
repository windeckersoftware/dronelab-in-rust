//! Rigid-body multirotor dynamics with motors, battery, drag and ground.

use dronelab_core::math::{Quat, Vec3, exp, sqrt};
use dronelab_core::{G, KinematicState, MAX_ROTORS};

use crate::airframe::Airframe;
use crate::battery::Battery;
use crate::rng::Rng;
use crate::sensors::{Imu, Wind};

const AIR_DENSITY: f32 = 1.225;

/// Which physical effects are simulated. [`Realism::IDEAL`] is a plain
/// rigid body with instant motors; [`Realism::FULL`] adds what separates
/// simulation from real flight. For gym-pybullet-drones' own models, use
/// `dronelab-compat`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Realism {
    /// First-order rotor spin-up/down instead of instant speed changes.
    pub motor_lag: bool,
    /// Battery voltage sag and discharge limit top rotor speed.
    pub battery: bool,
    /// Thrust falls with axial inflow (climb, forward flight through the disc).
    pub thrust_curve: bool,
    pub drag: bool,
    /// Gyro and accelerometer noise and drifting bias.
    pub imu_noise: bool,
    pub wind: bool,
}

impl Realism {
    pub const IDEAL: Realism = Realism {
        motor_lag: false,
        battery: false,
        thrust_curve: false,
        drag: false,
        imu_noise: false,
        wind: false,
    };
    pub const FULL: Realism = Realism {
        motor_lag: true,
        battery: true,
        thrust_curve: true,
        drag: true,
        imu_noise: true,
        wind: true,
    };
}

impl Default for Realism {
    fn default() -> Self {
        Realism::FULL
    }
}

/// The true vehicle after domain randomization. The flight code only knows
/// the nominal [`Airframe`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Physical {
    pub mass: f32,
    pub inertia: Vec3,
    /// Centre-of-mass offset from the nominal origin, body frame, m.
    pub com: Vec3,
    /// Thrust coefficient per rotor, N/(rad/s)^2.
    pub kf: [f32; MAX_ROTORS],
    /// Torque coefficient per rotor, N m/(rad/s)^2.
    pub km: [f32; MAX_ROTORS],
    pub motor_tau_up: f32,
    pub motor_tau_down: f32,
}

impl Physical {
    pub fn nominal(af: &Airframe) -> Self {
        let mut kf = [0.0; MAX_ROTORS];
        let mut km = [0.0; MAX_ROTORS];
        for (i, r) in af.vehicle.rotors().iter().enumerate() {
            kf[i] = af.kf(i);
            km[i] = kf[i] * r.torque_ratio;
        }
        Physical {
            mass: af.vehicle.mass,
            inertia: af.vehicle.inertia,
            com: Vec3::ZERO,
            kf,
            km,
            motor_tau_up: af.motor_tau_up,
            motor_tau_down: af.motor_tau_down,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct RigidState {
    pub pos: Vec3,
    pub vel: Vec3,
    pub quat: Quat,
    /// Body frame, rad/s.
    pub ang_vel: Vec3,
}

#[derive(Debug, Clone)]
pub struct Drone {
    pub phys: Physical,
    pub state: RigidState,
    /// Rotor speeds, rad/s.
    pub omega: [f32; MAX_ROTORS],
    /// Rotor thrusts from the last step, N.
    pub thrust: [f32; MAX_ROTORS],
    pub battery: Battery,
    pub wind: Wind,
    pub imu: Imu,
    pub on_ground: bool,
    pub crashed: bool,
    /// Simulated time, in physics steps.
    pub steps: u64,
}

impl Drone {
    pub fn new(
        af: &Airframe,
        phys: Physical,
        state: RigidState,
        battery: Battery,
        wind: Wind,
        imu: Imu,
    ) -> Self {
        // Start with rotors at hover speed so a reset in the air is steady.
        let w = af.omega_max * sqrt(af.hover_fraction() * af.vehicle.mass.recip() * phys.mass);
        let mut omega = [0.0; MAX_ROTORS];
        omega[..af.vehicle.rotor_count()].fill(w);
        Drone {
            phys,
            state,
            omega,
            thrust: [0.0; MAX_ROTORS],
            battery,
            wind,
            imu,
            on_ground: false,
            crashed: false,
            steps: 0,
        }
    }

    /// State as the flight code sees it: true pose, measured body rates.
    pub fn measured_state(&self) -> KinematicState {
        KinematicState {
            pos: self.state.pos,
            quat: self.state.quat,
            vel: self.state.vel,
            ang_vel: self.imu.last.gyro,
        }
    }

    /// Cosine of the tilt angle (1 = level, < 0 = upside down).
    pub fn up_dot(&self) -> f32 {
        self.state.quat.rotate(Vec3::Z).z
    }

    /// Advances by `dt` with each rotor commanded to `cmd[i]` of its nominal
    /// maximum thrust (0..1). The ESC maps the command to rotor speed
    /// assuming nominal props, so randomized props give different thrust.
    pub fn step(&mut self, af: &Airframe, realism: &Realism, cmd: &[f32], dt: f32, rng: &mut Rng) {
        let n = af.vehicle.rotor_count();
        let p = &self.phys;
        let s = self.state;

        let speed_limit = if realism.battery {
            af.omega_max * self.battery.voltage_ratio()
        } else {
            f32::INFINITY
        };
        for (w, &c) in self.omega[..n].iter_mut().zip(cmd) {
            let target = (af.omega_max * sqrt(c.clamp(0.0, 1.0))).min(speed_limit);
            *w = if realism.motor_lag {
                let tau = if target > *w {
                    p.motor_tau_up
                } else {
                    p.motor_tau_down
                };
                *w + (target - *w) * (1.0 - exp(-dt / tau))
            } else {
                target
            };
        }

        let rot = s.quat.to_mat();
        let z_b = rot.col(2);
        let wind = if realism.wind {
            self.wind.velocity()
        } else {
            Vec3::ZERO
        };
        let mut force = Vec3::ZERO; // body frame
        let mut torque = Vec3::ZERO; // body frame, about the centre of mass
        let mut shaft_power = 0.0;
        for (i, rotor) in af.vehicle.rotors().iter().enumerate() {
            let w = self.omega[i];
            let r = rotor.position - p.com;
            let mut t = p.kf[i] * w * w;
            if realism.thrust_curve && w > 1e-3 {
                let v_rotor = s.vel + rot.mul_vec(s.ang_vel.cross(r)) - wind;
                t *= inflow_factor(t, w, v_rotor, z_b, af);
            }
            let q = p.km[i] * w * w;
            self.thrust[i] = t;
            force.z += t;
            torque += r.cross(Vec3::new(0.0, 0.0, t)) + Vec3::new(0.0, 0.0, -rotor.spin * q);
            shaft_power += q * w;
        }
        if realism.battery {
            self.battery.draw(shaft_power / af.motor_efficiency, dt);
        }
        if realism.drag {
            let v_rel = rot.transpose().mul_vec(s.vel - wind);
            force += af.drag_area.mul_elem(v_rel) * (-0.5 * AIR_DENSITY * v_rel.norm());
        }

        let acc = rot.mul_vec(force) * (1.0 / p.mass) - Vec3::Z * G;
        let iw = p.inertia.mul_elem(s.ang_vel);
        let alpha = torque - s.ang_vel.cross(iw);
        let alpha = Vec3::new(
            alpha.x / p.inertia.x,
            alpha.y / p.inertia.y,
            alpha.z / p.inertia.z,
        );

        let mut next = s;
        next.vel += acc * dt;
        next.pos += next.vel * dt;
        next.ang_vel += alpha * dt;
        next.quat = s.quat.integrate(next.ang_vel, dt);

        self.on_ground = false;
        if next.pos.z < af.gear_height {
            let tilted = next.quat.rotate(Vec3::Z).z < 0.5;
            if next.vel.z < -af.crash_speed || tilted {
                self.crashed = true;
            }
            next.pos.z = af.gear_height;
            next.vel.z = next.vel.z.max(0.0);
            let k = 1.0 - (8.0 * dt).min(1.0);
            next.vel.x *= k;
            next.vel.y *= k;
            next.ang_vel = next.ang_vel * k;
            self.on_ground = true;
        }
        if !(next.pos.norm_squared().is_finite() && next.ang_vel.norm_squared().is_finite()) {
            self.crashed = true;
            next = s;
        }

        // Specific force for the accelerometer, after contact forces.
        let specific = next
            .quat
            .inverse_rotate((next.vel - s.vel) * (1.0 / dt) + Vec3::Z * G);
        self.state = next;
        if realism.wind {
            self.wind.step(dt, rng);
        }
        self.imu.sample(next.ang_vel, specific, dt, rng);
        self.steps += 1;
    }
}

/// Thrust relative to static thrust `t0` for a fixed-pitch prop whose hub
/// moves at `v_rotor` through the air (the prop thrust curve).
///
/// Blade element: thrust falls linearly with the inflow through the disc and
/// vanishes when it reaches the pitch speed `omega * pitch_length`.
/// Momentum theory gives the induced part of that inflow, which shrinks in
/// climb and in edgewise flight (translational lift), so forward flight is
/// not penalised the way pure axial climb is.
fn inflow_factor(t0: f32, omega: f32, v_rotor: Vec3, axis: Vec3, af: &Airframe) -> f32 {
    let v_ax = v_rotor.dot(axis);
    let v_edge2 = (v_rotor.norm_squared() - v_ax * v_ax).max(0.0);
    let area = core::f32::consts::PI * af.prop_radius * af.prop_radius;
    let vh2 = t0 / (2.0 * AIR_DENSITY * area);
    let vh = sqrt(vh2);
    // Fixed-point solve of v_i = v_h^2 / |(v_edge, v_ax + v_i)|, damped and
    // kept away from the vortex-ring singularity in steep descent.
    let mut vi = vh;
    for _ in 0..3 {
        let denom = sqrt(v_edge2 + (v_ax + vi) * (v_ax + vi)).max(0.5 * vh);
        vi = 0.5 * (vi + vh2 / denom);
    }
    (1.0 - (v_ax + vi - vh) / (omega * af.pitch_length)).clamp(0.0, 1.3)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::battery::Battery;

    fn drone(af: &Airframe, z: f32) -> Drone {
        let state = RigidState {
            pos: Vec3::new(0.0, 0.0, z),
            ..Default::default()
        };
        Drone::new(
            af,
            Physical::nominal(af),
            state,
            Battery::new(af.battery, 1.0),
            Wind::CALM,
            Imu::ideal(),
        )
    }

    #[test]
    fn hover_command_holds_altitude_ideal() {
        for name in Airframe::PRESETS {
            let af = Airframe::preset(name).unwrap();
            let mut d = drone(&af, 5.0);
            let cmd = vec![af.hover_fraction(); af.vehicle.rotor_count()];
            let mut rng = Rng::new(0);
            for _ in 0..240 {
                d.step(&af, &Realism::IDEAL, &cmd, 1.0 / 240.0, &mut rng);
            }
            assert!(
                (d.state.pos.z - 5.0).abs() < 1e-2,
                "{name}: z = {}",
                d.state.pos.z
            );
            assert!(
                d.state.ang_vel.norm() < 1e-3,
                "{name}: {:?}",
                d.state.ang_vel
            );
        }
    }

    #[test]
    fn free_fall_matches_gravity() {
        let af = Airframe::evtol_single_seat();
        let mut d = drone(&af, 100.0);
        let cmd = vec![0.0; 8];
        let mut rng = Rng::new(0);
        for _ in 0..240 {
            d.step(&af, &Realism::IDEAL, &cmd, 1.0 / 240.0, &mut rng);
        }
        assert!((d.state.vel.z + G).abs() < 1e-3, "{}", d.state.vel.z);
    }

    #[test]
    fn motor_lag_slows_spin_up() {
        let af = Airframe::evtol_two_seat();
        let mut d = drone(&af, 50.0);
        let w0 = d.omega[0];
        let cmd = vec![1.0; 16];
        let mut rng = Rng::new(0);
        let lagged = Realism {
            motor_lag: true,
            ..Realism::IDEAL
        };
        d.step(&af, &lagged, &cmd, 0.01, &mut rng);
        let frac = (d.omega[0] - w0) / (af.omega_max - w0);
        assert!((frac - (1.0 - libm::expf(-0.01 / af.motor_tau_up))).abs() < 1e-4);
    }

    #[test]
    fn hard_landing_is_a_crash() {
        let af = Airframe::evtol_single_seat();
        let mut d = drone(&af, 1.0);
        d.state.vel.z = -6.0;
        let mut rng = Rng::new(0);
        for _ in 0..60 {
            d.step(&af, &Realism::IDEAL, &[0.0; 8], 1.0 / 240.0, &mut rng);
        }
        assert!(d.crashed);
    }

    #[test]
    fn full_realism_stays_finite_at_full_throttle() {
        let af = Airframe::evtol_single_seat();
        let mut d = drone(&af, 10.0);
        let mut rng = Rng::new(9);
        for _ in 0..2400 {
            d.step(&af, &Realism::FULL, &[1.0; 8], 1.0 / 240.0, &mut rng);
        }
        assert!(d.state.pos.z.is_finite() && d.state.pos.z > 10.0);
        // Climbing fast: the thrust curve and drag cap the climb rate.
        assert!(d.state.vel.z < 40.0, "{}", d.state.vel.z);
        assert!(d.battery.voltage < af.battery.open_circuit_voltage(1.0));
    }

    #[test]
    fn thrust_curve_shapes() {
        let af = Airframe::evtol_single_seat();
        let w = 190.0;
        let t0 = af.kf(0) * w * w;
        let f = |v: Vec3| inflow_factor(t0, w, v, Vec3::Z, &af);
        assert!((f(Vec3::ZERO) - 1.0).abs() < 1e-3, "hover");
        assert!(f(Vec3::new(0.0, 0.0, 3.0)) < 0.97, "climb loses thrust");
        assert!(f(Vec3::new(0.0, 0.0, -2.0)) > 1.0, "descent gains thrust");
        assert!(f(Vec3::new(12.0, 0.0, 0.0)) > 1.05, "translational lift");
        assert!(f(Vec3::new(0.0, 0.0, -50.0)).is_finite());
    }
}

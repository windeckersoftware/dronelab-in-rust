//! Cascaded PID flight controller, after gym-pybullet-drones' `DSLPIDControl`.
//!
//! Same structure as the Python controller (PID on position giving a thrust
//! vector, then PID on the rotation-matrix attitude error), but the gains are
//! normalised by mass and inertia: position gains give m/s^2 and attitude
//! gains give rad/s^2. That makes one tuning rule work from a 27 g
//! Crazyflie to a 600 kg eVTOL, and the mixer replaces the fixed
//! Crazyflie PWM mixer.

use crate::control::{BaseControl, Control, ControlOutput, KinematicState, Target};
use crate::math::{Mat3, Vec3, cos, sin, sqrt, tan, wrap_angle};
use crate::mixer::Mixer;
use crate::vehicle::VehicleParams;

/// PID gains. Position gains give m/s^2 from position error (P), velocity
/// error (D) and the integral of velocity error (I); attitude gains give
/// rad/s^2 from rad, rad/s and rad*s.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PidGains {
    pub p_pos: Vec3,
    pub i_pos: Vec3,
    pub d_pos: Vec3,
    pub p_att: Vec3,
    pub i_att: Vec3,
    pub d_att: Vec3,
}

/// New PID coefficients; `None` keeps the current value. Named like the
/// arguments of Python's `setPIDCoefficients`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PidCoefficients {
    pub p_coeff_pos: Option<Vec3>,
    pub i_coeff_pos: Option<Vec3>,
    pub d_coeff_pos: Option<Vec3>,
    pub p_coeff_att: Option<Vec3>,
    pub i_coeff_att: Option<Vec3>,
    pub d_coeff_att: Option<Vec3>,
}

const ZETA: f32 = 0.9;
/// Attitude error beyond which the attitude integral stops growing faster, rad.
const ATT_INTEGRAL_BAND: f32 = 0.3;

impl PidGains {
    /// Gains placing the attitude loop at `att_wn` and the position loop at
    /// `pos_wn` rad/s, both with damping 0.9. Yaw and the vertical axis get
    /// their own ratios because yaw torque is weak and climb is direct.
    pub fn from_bandwidth(att_wn: f32, pos_wn: f32) -> Self {
        let (a, ay) = (att_wn, 0.5 * att_wn);
        let (p, pz) = (pos_wn, 1.5 * pos_wn);
        PidGains {
            p_pos: Vec3::new(p * p, p * p, pz * pz),
            i_pos: Vec3::new(0.4 * p * p, 0.4 * p * p, 0.4 * pz * pz),
            d_pos: Vec3::new(2.0 * ZETA * p, 2.0 * ZETA * p, 2.0 * ZETA * pz),
            p_att: Vec3::new(a * a, a * a, ay * ay),
            i_att: Vec3::new(0.25 * a * a * a, 0.25 * a * a * a, 0.1 * ay * ay * ay),
            d_att: Vec3::new(2.0 * ZETA * a, 2.0 * ZETA * a, 2.0 * ZETA * ay),
        }
    }

    /// Default gains for a vehicle, from its motor time constant: the
    /// attitude loop runs well below the motor bandwidth, and the position
    /// loop well below the attitude loop.
    pub fn for_vehicle(v: &VehicleParams) -> Self {
        let att_wn = 1.0 / (3.0 * v.motor_tau);
        PidGains::from_bandwidth(att_wn, att_wn / 5.0)
    }

    /// Overwrites each gain for which `coeffs` has a value.
    pub fn apply(&mut self, coeffs: &PidCoefficients) {
        let set = |gain: &mut Vec3, new: Option<Vec3>| {
            if let Some(v) = new {
                *gain = v;
            }
        };
        set(&mut self.p_pos, coeffs.p_coeff_pos);
        set(&mut self.i_pos, coeffs.i_coeff_pos);
        set(&mut self.d_pos, coeffs.d_coeff_pos);
        set(&mut self.p_att, coeffs.p_coeff_att);
        set(&mut self.i_att, coeffs.i_coeff_att);
        set(&mut self.d_att, coeffs.d_coeff_att);
    }
}

/// Integral state bound so that `gain * state` stays within `limit`.
fn state_limit(limit: Vec3, gain: Vec3) -> Vec3 {
    let f = |l: f32, k: f32| if k > 0.0 { l / k } else { 0.0 };
    Vec3::new(f(limit.x, gain.x), f(limit.y, gain.y), f(limit.z, gain.z))
}

/// Position and attitude PID controller for any multirotor.
#[derive(Debug, Clone, Copy)]
pub struct PidControl {
    pub base: BaseControl,
    pub gains: PidGains,
    pub mixer: Mixer,
    /// Largest correction the position integral may add, m/s^2. The z
    /// default covers a vehicle about 30% heavier than its nominal mass.
    pub integral_pos_limit: Vec3,
    /// Largest correction the attitude integral may add, rad/s^2 (e.g. for
    /// an off-centre passenger).
    pub integral_att_limit: Vec3,
    integral_pos: Vec3,
    integral_att: Vec3,
    tilt_limited: bool,
    mixer_saturated: bool,
}

impl PidControl {
    /// Returns `None` if the vehicle's rotor layout cannot be mixed.
    pub fn new(vehicle: VehicleParams, g: f32) -> Option<Self> {
        Self::with_gains(vehicle, g, PidGains::for_vehicle(&vehicle))
    }

    pub fn with_gains(vehicle: VehicleParams, g: f32, gains: PidGains) -> Option<Self> {
        Some(PidControl {
            base: BaseControl::new(vehicle, g),
            gains,
            mixer: Mixer::new(&vehicle)?,
            integral_pos_limit: Vec3::new(0.2 * g, 0.2 * g, 0.3 * g),
            integral_att_limit: Self::default_att_limit(&vehicle),
            integral_pos: Vec3::ZERO,
            integral_att: Vec3::ZERO,
            tilt_limited: false,
            mixer_saturated: false,
        })
    }

    /// Room for the attitude integral: a torque bias of 15% of weight times
    /// rotor radius (off-centre mass plus uneven motors), as angular
    /// acceleration.
    fn default_att_limit(v: &VehicleParams) -> Vec3 {
        let arm = v
            .rotors()
            .iter()
            .map(|r| sqrt(r.position.x * r.position.x + r.position.y * r.position.y))
            .fold(0.0, f32::max);
        let torque = 0.15 * arm * v.weight();
        Vec3::new(
            torque / v.inertia.x,
            torque / v.inertia.y,
            0.1 * torque / v.inertia.z,
        )
    }

    /// Desired acceleration from the position loop, before the tilt limit.
    ///
    /// Without the integral and unsaturated, this is exactly the Python form
    /// `P*pos_e + D*vel_e`; the speed limits enter by clamping the velocity
    /// the P term asks for.
    fn position_accel(&mut self, dt: f32, cur: &KinematicState, target: &Target) -> (Vec3, Vec3) {
        let v = &self.base.vehicle;
        let g = &self.gains;
        let pos_e = target.pos - cur.pos;

        let ratio = |p: f32, d: f32| if d > 0.0 { p / d } else { 0.0 };
        let mut v_des = target.vel
            + Vec3::new(
                ratio(g.p_pos.x, g.d_pos.x) * pos_e.x,
                ratio(g.p_pos.y, g.d_pos.y) * pos_e.y,
                ratio(g.p_pos.z, g.d_pos.z) * pos_e.z,
            );
        let h = sqrt(v_des.x * v_des.x + v_des.y * v_des.y);
        let xy_saturated = h > v.max_speed_xy || self.tilt_limited;
        if h > v.max_speed_xy {
            let k = v.max_speed_xy / h;
            v_des.x *= k;
            v_des.y *= k;
        }
        let z_saturated = v_des.z.abs() > v.max_speed_z;
        v_des.z = v_des.z.clamp(-v.max_speed_z, v.max_speed_z);

        // The integral acts on the velocity-loop error, as in PX4: in cruise
        // the vehicle tracks `v_des`, so it only learns persistent offsets
        // (extra mass, wind). Saturated axes do not integrate (anti-windup).
        let vel_e = v_des - cur.vel;
        let mut step = vel_e * dt;
        if xy_saturated {
            step.x = 0.0;
            step.y = 0.0;
        }
        if z_saturated {
            step.z = 0.0;
        }
        self.integral_pos =
            (self.integral_pos + step).clamp_abs(state_limit(self.integral_pos_limit, g.i_pos));

        let acc = g.d_pos.mul_elem(vel_e) + g.i_pos.mul_elem(self.integral_pos);
        (acc, pos_e)
    }

    /// Attitude loop: desired rotation to body torque.
    fn attitude_torque(
        &mut self,
        dt: f32,
        cur: &KinematicState,
        rot: &Mat3,
        target_rot: &Mat3,
        rates: Vec3,
    ) -> Vec3 {
        let v = &self.base.vehicle;
        let g = &self.gains;
        let rot_e = target_rot
            .transpose()
            .mul_mat(rot)
            .sub_mat(&rot.transpose().mul_mat(target_rot))
            .vee()
            * 0.5;
        let rate_e = cur.ang_vel - rates;
        // Integrate a clamped error, and nothing while the rotors are
        // saturated, so manoeuvres do not wind the integral up while a
        // steady offset (e.g. off-centre mass) still trims out.
        let step = if self.mixer_saturated {
            Vec3::ZERO
        } else {
            rot_e.clamp_abs(Vec3::splat(ATT_INTEGRAL_BAND)) * dt
        };
        self.integral_att =
            (self.integral_att - step).clamp_abs(state_limit(self.integral_att_limit, g.i_att));
        let alpha = -g.p_att.mul_elem(rot_e) - g.d_att.mul_elem(rate_e)
            + g.i_att.mul_elem(self.integral_att);
        let iw = v.inertia.mul_elem(cur.ang_vel);
        v.inertia.mul_elem(alpha) + cur.ang_vel.cross(iw)
    }
}

impl Control for PidControl {
    fn base(&self) -> &BaseControl {
        &self.base
    }

    fn base_mut(&mut self) -> &mut BaseControl {
        &mut self.base
    }

    fn compute_control(&mut self, dt: f32, cur: &KinematicState, target: &Target) -> ControlOutput {
        self.base.control_counter = self.base.control_counter.wrapping_add(1);
        let (mut acc, pos_e) = self.position_accel(dt, cur, target);
        let v = self.base.vehicle;
        let g0 = self.base.g;

        // Keep at least 30% of gravity's worth of lift, then limit tilt.
        acc.z = acc.z.max(-0.7 * g0);
        let vertical = acc.z + g0;
        let h = sqrt(acc.x * acc.x + acc.y * acc.y);
        let h_max = vertical * tan(v.max_tilt);
        self.tilt_limited = h > h_max;
        if self.tilt_limited {
            acc.x *= h_max / h;
            acc.y *= h_max / h;
        }
        let force = Vec3::new(acc.x, acc.y, vertical) * v.mass;

        let rot = cur.quat.to_mat();
        let collective = force.dot(rot.col(2)).max(0.0);
        let yaw = target.rpy.z;
        let z_d = force.normalized_or(Vec3::Z);
        let y_d = z_d
            .cross(Vec3::new(cos(yaw), sin(yaw), 0.0))
            .normalized_or(Vec3::Y);
        let x_d = y_d.cross(z_d);
        let target_rot = Mat3::from_cols(x_d, y_d, z_d);

        let torque = self.attitude_torque(dt, cur, &rot, &target_rot, target.rpy_rates);
        let mix = self.mixer.mix(collective, torque);
        self.mixer_saturated = mix.saturated;
        ControlOutput {
            thrust: mix.thrust,
            rotor_count: v.rotor_count(),
            pos_error: pos_e,
            yaw_error: wrap_angle(yaw - cur.quat.to_rpy().z),
            saturated: mix.saturated,
        }
    }

    fn reset(&mut self) {
        self.base.reset();
        self.integral_pos = Vec3::ZERO;
        self.integral_att = Vec3::ZERO;
        self.tilt_limited = false;
        self.mixer_saturated = false;
    }

    fn pid_gains_mut(&mut self) -> Option<&mut PidGains> {
        Some(&mut self.gains)
    }
}

/// Body-rate controller: collective thrust and body rates to rotor thrust.
/// This is the inner loop of [`PidControl`], exposed for "CTBR" actions
/// (collective thrust and body rates), the usual action space for learned
/// racing policies.
#[derive(Debug, Clone, Copy)]
pub struct RateControl {
    pub vehicle: VehicleParams,
    pub mixer: Mixer,
    /// Rate gains, rad/s^2 per rad/s.
    pub k_rate: Vec3,
}

impl RateControl {
    pub fn new(vehicle: VehicleParams) -> Option<Self> {
        let k_rate = PidGains::for_vehicle(&vehicle).d_att * 1.5;
        Some(RateControl {
            vehicle,
            mixer: Mixer::new(&vehicle)?,
            k_rate,
        })
    }

    /// Rotor thrusts for `collective` newtons and body rates `rates`.
    pub fn compute(&self, collective: f32, rates: Vec3, ang_vel: Vec3) -> crate::mixer::MixOutput {
        let i = self.vehicle.inertia;
        let alpha = self.k_rate.mul_elem(rates - ang_vel);
        let torque = i.mul_elem(alpha) + ang_vel.cross(i.mul_elem(ang_vel));
        self.mixer.mix(collective, torque)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::G;

    #[test]
    fn hovering_at_target_asks_for_weight() {
        for name in VehicleParams::PRESETS {
            let v = VehicleParams::preset(name).unwrap();
            let mut c = PidControl::new(v, G).unwrap();
            let target = Target::at(Vec3::new(0.0, 0.0, 1.0));
            let cur = KinematicState {
                pos: target.pos,
                ..Default::default()
            };
            let out = c.compute_control(0.02, &cur, &target);
            let total: f32 = out.thrust().iter().sum();
            assert!(
                (total - v.weight()).abs() < 1e-3 * v.weight(),
                "{name}: {total}"
            );
            assert!(!out.saturated);
        }
    }

    #[test]
    fn tilts_towards_a_target_ahead() {
        let v = VehicleParams::evtol_single_seat();
        let mut c = PidControl::new(v, G).unwrap();
        let cur = KinematicState::default();
        let out = c.compute_control(0.02, &cur, &Target::at(Vec3::new(50.0, 0.0, 0.0)));
        let (_, tau) = Mixer::wrench(&v, out.thrust());
        // Pitching nose-down to fly +x is a positive torque about body y.
        assert!(tau.y > 0.0, "{tau:?}");
    }

    #[test]
    fn set_pid_coefficients_keeps_unset_gains() {
        let mut c = PidControl::new(VehicleParams::crazyflie_x(), G).unwrap();
        let before = c.gains;
        let new = Vec3::new(0.4, 0.4, 1.25);
        c.set_pid_coefficients(&PidCoefficients {
            p_coeff_pos: Some(new),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(c.gains.p_pos, new);
        assert_eq!(c.gains.d_att, before.d_att);
    }

    #[test]
    fn rate_control_damps_spin() {
        let v = VehicleParams::evtol_two_seat();
        let rc = RateControl::new(v).unwrap();
        let out = rc.compute(v.weight(), Vec3::ZERO, Vec3::new(0.0, 0.0, 0.5));
        let (_, tau) = Mixer::wrench(&v, &out.thrust[..v.rotor_count()]);
        assert!(tau.z < 0.0, "{tau:?}");
    }
}

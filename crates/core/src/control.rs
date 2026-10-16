//! Port of gym-pybullet-drones' `control/BaseControl.py`, generalised from
//! four Crazyflie motors to any rotor layout.
//!
//! Python's `BaseControl` class is split in two:
//! - [`BaseControl`], a struct with the shared constants and counter, which
//!   concrete controllers embed;
//! - [`Control`], a trait whose required method is `compute_control` and whose
//!   provided methods are `reset`, `compute_control_from_state` and
//!   `set_pid_coefficients`.
//!
//! Controllers output thrust per rotor in newtons rather than Crazyflie RPMs,
//! so the same code drives a 27 g quadcopter and a 600 kg eVTOL.

use core::fmt;

use crate::MAX_ROTORS;
use crate::math::{Quat, Vec3};
use crate::pid::{PidCoefficients, PidGains};
use crate::vehicle::VehicleParams;

/// Shared state of every controller.
#[derive(Debug, Clone, Copy)]
pub struct BaseControl {
    /// The vehicle to control.
    pub vehicle: VehicleParams,
    /// Gravitational acceleration, m/s^2.
    pub g: f32,
    /// The gravitational force (M*g) acting on the vehicle, N.
    pub gravity: f32,
    /// General use counter, set to zero by [`BaseControl::reset`].
    pub control_counter: u32,
}

impl BaseControl {
    pub fn new(vehicle: VehicleParams, g: f32) -> Self {
        BaseControl {
            vehicle,
            g,
            gravity: g * vehicle.mass,
            control_counter: 0,
        }
    }

    /// Resets the general use counter to zero.
    pub fn reset(&mut self) {
        self.control_counter = 0;
    }
}

/// The vehicle's current state, as the flight code sees it.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct KinematicState {
    /// Position, world frame, m.
    pub pos: Vec3,
    /// Orientation, body to world.
    pub quat: Quat,
    /// Velocity, world frame, m/s.
    pub vel: Vec3,
    /// Angular velocity, body frame (what a gyro measures), rad/s.
    pub ang_vel: Vec3,
}

impl KinematicState {
    /// Extracts the state from the (20,)-shaped array that gym-pybullet-drones'
    /// `BaseAviary.step()` returns under the key `"state"`. That array holds
    /// a world-frame angular velocity, which is rotated into the body frame.
    pub fn from_pybullet_state(state: &[f32; 20]) -> Self {
        let v = |i: usize| Vec3::new(state[i], state[i + 1], state[i + 2]);
        let quat = Quat::from_xyzw([state[3], state[4], state[5], state[6]]);
        KinematicState {
            pos: v(0),
            quat,
            vel: v(10),
            ang_vel: quat.inverse_rotate(v(13)),
        }
    }
}

/// The desired state. Everything but `pos` defaults to zero, as in Python.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Target {
    /// Desired position, m.
    pub pos: Vec3,
    /// Desired orientation as roll, pitch, yaw, rad. Position control
    /// decides roll and pitch; only yaw is used by [`crate::PidControl`].
    pub rpy: Vec3,
    /// Desired velocity, m/s.
    pub vel: Vec3,
    /// Desired body rates, rad/s.
    pub rpy_rates: Vec3,
}

impl Target {
    /// A target at `pos` with zero orientation, velocity and rates.
    pub fn at(pos: Vec3) -> Self {
        Target {
            pos,
            ..Default::default()
        }
    }
}

/// What a controller returns for one step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ControlOutput {
    /// Thrust per rotor, N. Only the first `rotor_count` entries are used.
    pub thrust: [f32; MAX_ROTORS],
    pub rotor_count: usize,
    /// Current position error, m.
    pub pos_error: Vec3,
    /// Current yaw error, rad.
    pub yaw_error: f32,
    /// The mixer could not meet the request within rotor limits.
    pub saturated: bool,
}

impl ControlOutput {
    pub fn thrust(&self) -> &[f32] {
        &self.thrust[..self.rotor_count]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlError {
    /// `set_pid_coefficients` was called on a controller without PID gains.
    NotPid,
}

impl fmt::Display for ControlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ControlError::NotPid => f.write_str("controller has no PID coefficients to set"),
        }
    }
}

impl core::error::Error for ControlError {}

/// Interface of every controller.
pub trait Control {
    /// The embedded shared state.
    fn base(&self) -> &BaseControl;
    fn base_mut(&mut self) -> &mut BaseControl;

    /// Computes the control action for a single vehicle.
    fn compute_control(
        &mut self,
        control_timestep: f32,
        cur: &KinematicState,
        target: &Target,
    ) -> ControlOutput;

    /// Resets the controller. Controllers with more state (e.g. PID
    /// integrals) override this and call `self.base_mut().reset()`.
    fn reset(&mut self) {
        self.base_mut().reset();
    }

    /// Computes the control action from gym-pybullet-drones' 20-value state.
    fn compute_control_from_state(
        &mut self,
        control_timestep: f32,
        state: &[f32; 20],
        target: &Target,
    ) -> ControlOutput {
        self.compute_control(
            control_timestep,
            &KinematicState::from_pybullet_state(state),
            target,
        )
    }

    /// The controller's PID gains, if it has any. PID controllers override
    /// this to enable [`Control::set_pid_coefficients`].
    fn pid_gains_mut(&mut self) -> Option<&mut PidGains> {
        None
    }

    /// Sets the coefficients of a PID controller, keeping those left `None`.
    ///
    /// Fails with [`ControlError::NotPid`] if the controller has no PID gains.
    fn set_pid_coefficients(&mut self, coeffs: &PidCoefficients) -> Result<(), ControlError> {
        let gains = self.pid_gains_mut().ok_or(ControlError::NotPid)?;
        gains.apply(coeffs);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Echo {
        base: BaseControl,
        last: Option<(KinematicState, Target)>,
    }

    impl Control for Echo {
        fn base(&self) -> &BaseControl {
            &self.base
        }
        fn base_mut(&mut self) -> &mut BaseControl {
            &mut self.base
        }
        fn compute_control(
            &mut self,
            _dt: f32,
            cur: &KinematicState,
            target: &Target,
        ) -> ControlOutput {
            self.base.control_counter += 1;
            self.last = Some((*cur, *target));
            ControlOutput {
                thrust: [0.0; MAX_ROTORS],
                rotor_count: 0,
                pos_error: Vec3::ZERO,
                yaw_error: 0.0,
                saturated: false,
            }
        }
    }

    fn echo() -> Echo {
        Echo {
            base: BaseControl::new(VehicleParams::crazyflie_x(), 9.8),
            last: None,
        }
    }

    #[test]
    fn base_matches_python_gravity() {
        let b = echo().base;
        assert!((b.gravity - 9.8 * 0.027).abs() < 1e-6);
    }

    #[test]
    fn from_state_slices_like_python() {
        let mut state = [0.0f32; 20];
        for (i, s) in state.iter_mut().enumerate() {
            *s = i as f32;
        }
        state[3..7].copy_from_slice(&[0.0, 0.0, 0.0, 1.0]); // identity quaternion
        let mut c = echo();
        c.compute_control_from_state(1.0 / 48.0, &state, &Target::at(Vec3::new(1.0, 2.0, 3.0)));
        let (cur, target) = c.last.unwrap();
        assert_eq!(cur.pos, Vec3::new(0.0, 1.0, 2.0));
        assert_eq!(cur.vel, Vec3::new(10.0, 11.0, 12.0));
        assert_eq!(cur.ang_vel, Vec3::new(13.0, 14.0, 15.0));
        assert_eq!(target.vel, Vec3::ZERO);
    }

    #[test]
    fn reset_zeroes_counter() {
        let mut c = echo();
        c.compute_control(0.1, &KinematicState::default(), &Target::default());
        assert_eq!(c.base().control_counter, 1);
        c.reset();
        assert_eq!(c.base().control_counter, 0);
    }

    #[test]
    fn set_pid_coefficients_needs_pid_gains() {
        assert_eq!(
            echo().set_pid_coefficients(&PidCoefficients::default()),
            Err(ControlError::NotPid)
        );
    }
}

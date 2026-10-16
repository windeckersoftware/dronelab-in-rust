//! Vehicle description used by the controller and mixer.
//!
//! Frames: body x forward, y left, z up. Rotors push along body +z.

use crate::math::{Vec3, cos, sin};
use crate::{G, MAX_ROTORS};

/// One rotor as the flight code sees it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rotor {
    /// Hub position in the body frame, m.
    pub position: Vec3,
    /// +1 if the rotor turns counter-clockwise seen from above, -1 otherwise.
    /// A CCW rotor pushes the body clockwise (negative yaw torque).
    pub spin: f32,
    /// Reaction torque per newton of thrust (km / kf), m.
    pub torque_ratio: f32,
    /// Thrust at full command and nominal battery voltage, N.
    pub max_thrust: f32,
}

impl Rotor {
    pub const NONE: Rotor = Rotor {
        position: Vec3::ZERO,
        spin: 0.0,
        torque_ratio: 0.0,
        max_thrust: 0.0,
    };
}

/// Mass, inertia and rotor layout of a multirotor vehicle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VehicleParams {
    pub name: &'static str,
    /// Total mass including payload, kg.
    pub mass: f32,
    /// Principal moments of inertia about body x, y, z, kg m^2.
    pub inertia: Vec3,
    rotors: [Rotor; MAX_ROTORS],
    rotor_count: usize,
    /// Nominal motor+prop spin-up time constant, s. Large props are slow,
    /// which limits how fast the attitude loop can be.
    pub motor_tau: f32,
    /// Tilt limit the position controller respects, rad.
    pub max_tilt: f32,
    /// Horizontal speed limit for position control, m/s.
    pub max_speed_xy: f32,
    /// Vertical speed limit for position control, m/s.
    pub max_speed_z: f32,
}

impl VehicleParams {
    /// Builds a vehicle with small-drone defaults for the public tuning
    /// fields; panics if there are more than [`MAX_ROTORS`] rotors.
    pub fn new(name: &'static str, mass: f32, inertia: Vec3, rotors: &[Rotor]) -> Self {
        assert!(rotors.len() <= MAX_ROTORS, "too many rotors");
        let mut all = [Rotor::NONE; MAX_ROTORS];
        all[..rotors.len()].copy_from_slice(rotors);
        VehicleParams {
            name,
            mass,
            inertia,
            rotors: all,
            rotor_count: rotors.len(),
            motor_tau: 0.03,
            max_tilt: 0.6,
            max_speed_xy: 3.0,
            max_speed_z: 1.5,
        }
    }

    pub fn rotors(&self) -> &[Rotor] {
        &self.rotors[..self.rotor_count]
    }

    pub fn rotors_mut(&mut self) -> &mut [Rotor] {
        &mut self.rotors[..self.rotor_count]
    }

    pub fn rotor_count(&self) -> usize {
        self.rotor_count
    }

    /// Weight, N.
    pub fn weight(&self) -> f32 {
        self.mass * G
    }

    /// Sum of all rotors' maximum thrust, N.
    pub fn max_total_thrust(&self) -> f32 {
        self.rotors().iter().map(|r| r.max_thrust).sum()
    }

    pub fn thrust_to_weight(&self) -> f32 {
        self.max_total_thrust() / self.weight()
    }

    /// Bitcraze Crazyflie 2.0 in X configuration, from gym-pybullet-drones'
    /// `cf2x.urdf`. Useful for comparing against PyBullet.
    pub fn crazyflie_x() -> Self {
        let mass = 0.027;
        let max_thrust = 2.25 * mass * G / 4.0;
        let ratio = 7.94e-12 / 3.16e-10;
        let r = |x: f32, y: f32, spin: f32| Rotor {
            position: Vec3::new(x, y, 0.0),
            spin,
            torque_ratio: ratio,
            max_thrust,
        };
        let rotors = [
            r(0.028, -0.028, 1.0),
            r(-0.028, -0.028, -1.0),
            r(-0.028, 0.028, 1.0),
            r(0.028, 0.028, -1.0),
        ];
        VehicleParams::new(
            "crazyflie_x",
            mass,
            Vec3::new(1.4e-5, 1.4e-5, 2.17e-5),
            &rotors,
        )
    }

    /// Generic single-seat eVTOL: 4 arms with coaxial rotor pairs (X8),
    /// 200 kg with pilot. Numbers are illustrative, not a specific aircraft.
    pub fn evtol_single_seat() -> Self {
        let rotors = coaxial_ring::<8>(4, 1.25, core::f32::consts::FRAC_PI_4, 560.0, 470.0, 0.07);
        VehicleParams {
            motor_tau: 0.15,
            max_tilt: 0.44,
            max_speed_xy: 12.0,
            max_speed_z: 3.0,
            ..VehicleParams::new(
                "evtol_single_seat",
                200.0,
                Vec3::new(55.0, 55.0, 95.0),
                &rotors,
            )
        }
    }

    /// Generic two-seat eVTOL: 8 arms with coaxial rotor pairs (16 rotors),
    /// 620 kg loaded. Numbers are illustrative, not a specific aircraft.
    pub fn evtol_two_seat() -> Self {
        let rotors = coaxial_ring::<16>(8, 2.4, core::f32::consts::FRAC_PI_8, 780.0, 664.0, 0.08);
        VehicleParams {
            motor_tau: 0.2,
            max_tilt: 0.35,
            max_speed_xy: 15.0,
            max_speed_z: 3.0,
            ..VehicleParams::new(
                "evtol_two_seat",
                620.0,
                Vec3::new(420.0, 420.0, 780.0),
                &rotors,
            )
        }
    }

    /// Looks up a preset by name.
    pub fn preset(name: &str) -> Option<Self> {
        match name {
            "crazyflie_x" => Some(Self::crazyflie_x()),
            "evtol_single_seat" => Some(Self::evtol_single_seat()),
            "evtol_two_seat" => Some(Self::evtol_two_seat()),
            _ => None,
        }
    }

    pub const PRESETS: [&'static str; 3] = ["crazyflie_x", "evtol_single_seat", "evtol_two_seat"];
}

/// `arms` arms evenly spaced from `first_angle`, each with an upper and a
/// lower rotor turning opposite ways; neighbouring arms alternate.
/// The lower rotor works in the upper one's wake, so it makes less thrust.
fn coaxial_ring<const N: usize>(
    arms: usize,
    radius: f32,
    first_angle: f32,
    upper_thrust: f32,
    lower_thrust: f32,
    torque_ratio: f32,
) -> [Rotor; N] {
    assert_eq!(N, 2 * arms);
    let mut rotors = [Rotor::NONE; N];
    for k in 0..arms {
        let a = first_angle + k as f32 * 2.0 * core::f32::consts::PI / arms as f32;
        let (x, y) = (radius * cos(a), radius * sin(a));
        let spin = if k % 2 == 0 { 1.0 } else { -1.0 };
        rotors[2 * k] = Rotor {
            position: Vec3::new(x, y, 0.12),
            spin,
            torque_ratio,
            max_thrust: upper_thrust,
        };
        rotors[2 * k + 1] = Rotor {
            position: Vec3::new(x, y, -0.12),
            spin: -spin,
            torque_ratio,
            max_thrust: lower_thrust,
        };
    }
    rotors
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_can_lift_off_with_margin() {
        for name in VehicleParams::PRESETS {
            let v = VehicleParams::preset(name).unwrap();
            let tw = v.thrust_to_weight();
            assert!(tw > 1.8 && tw < 2.5, "{name}: thrust/weight {tw}");
            let spin_sum: f32 = v.rotors().iter().map(|r| r.spin).sum();
            assert_eq!(spin_sum, 0.0, "{name}: rotors must balance yaw");
        }
    }
}

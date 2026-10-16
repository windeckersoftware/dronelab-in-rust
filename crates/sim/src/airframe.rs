//! Physical description of a vehicle: what the flight code knows
//! ([`VehicleParams`]) plus what only the physics needs (powertrain,
//! battery, aerodynamics, landing gear).

use dronelab_core::math::Vec3;
use dronelab_core::{G, VehicleParams};

use crate::battery::BatterySpec;
use crate::enums::DroneModel;
use crate::urdf::{UrdfError, UrdfParam, get_urdf_parameter, rotor_positions};

#[derive(Debug, Clone)]
pub struct Airframe {
    pub vehicle: VehicleParams,
    /// Rotor speed at full command and nominal battery voltage, rad/s.
    pub omega_max: f32,
    /// Propeller pitch / 2*pi, m. Thrust falls to zero when the air comes
    /// through the disc at `omega * pitch_length` (the prop thrust curve).
    pub pitch_length: f32,
    /// Propeller radius, m.
    pub prop_radius: f32,
    /// Motor time constants for spinning up and down, s.
    pub motor_tau_up: f32,
    pub motor_tau_down: f32,
    /// Shaft power / electrical power.
    pub motor_efficiency: f32,
    pub battery: BatterySpec,
    /// Drag coefficient times area along body x, y, z, m^2.
    pub drag_area: Vec3,
    /// Height of the centre of mass above the ground when landed, m.
    pub gear_height: f32,
    /// Vertical speed at touchdown above which the landing is a crash, m/s.
    pub crash_speed: f32,
    /// Largest body-rate command for CTBR actions, rad/s.
    pub max_rate: f32,
}

impl Airframe {
    pub const PRESETS: [&'static str; 4] = [
        "crazyflie_x",
        "racer",
        "evtol_single_seat",
        "evtol_two_seat",
    ];

    pub fn preset(name: &str) -> Option<Self> {
        match name {
            "crazyflie_x" => Some(Self::crazyflie_x()),
            "racer" => Some(Self::racer()),
            "evtol_single_seat" => Some(Self::evtol_single_seat()),
            "evtol_two_seat" => Some(Self::evtol_two_seat()),
            _ => None,
        }
    }

    /// Thrust coefficient of rotor `i`, N/(rad/s)^2.
    pub fn kf(&self, i: usize) -> f32 {
        self.vehicle.rotors()[i].max_thrust / (self.omega_max * self.omega_max)
    }

    /// Distance from the centre to the outermost prop tip, m.
    pub fn radius(&self) -> f32 {
        let hub = self
            .vehicle
            .rotors()
            .iter()
            .map(|r| Vec3::new(r.position.x, r.position.y, 0.0).norm())
            .fold(0.0, f32::max);
        hub + self.prop_radius
    }

    /// Fraction of full thrust each rotor needs to hover.
    pub fn hover_fraction(&self) -> f32 {
        self.vehicle.weight() / self.vehicle.max_total_thrust()
    }

    /// Crazyflie 2.0 (X), read from gym-pybullet-drones' `cf2x.urdf`.
    pub fn crazyflie_x() -> Self {
        let mut a = Self::from_urdf(DroneModel::Cf2x).expect("bundled cf2x.urdf is valid");
        a.vehicle.name = "crazyflie_x";
        a
    }

    /// 5-inch-class racing quad, read from gym-pybullet-drones' `racer.urdf`.
    pub fn racer() -> Self {
        let mut a = Self::from_urdf(DroneModel::Race).expect("bundled racer.urdf is valid");
        a.vehicle.name = "racer";
        a.vehicle.max_tilt = 1.0;
        a.vehicle.max_speed_xy = 15.0;
        a.vehicle.max_speed_z = 5.0;
        a.vehicle.motor_tau = 0.02;
        a.motor_tau_up = 0.02;
        a.motor_tau_down = 0.03;
        a.pitch_length = 0.017;
        a.battery = BatterySpec {
            cells: 6,
            capacity_ah: 1.3,
            r_internal: 0.06,
            aux_power: 3.0,
        };
        a.drag_area = Vec3::new(0.008, 0.008, 0.015);
        a.gear_height = 0.03;
        a.crash_speed = 5.0;
        a.max_rate = 12.0;
        a
    }

    /// Generic single-seat eVTOL (X8 coaxial, 200 kg loaded, ~12 kWh pack).
    pub fn evtol_single_seat() -> Self {
        Airframe {
            vehicle: VehicleParams::evtol_single_seat(),
            omega_max: 272.0, // ~2600 rpm
            pitch_length: 0.143,
            prop_radius: 0.7,
            motor_tau_up: 0.15,
            motor_tau_down: 0.2,
            motor_efficiency: 0.85,
            battery: BatterySpec {
                cells: 96,
                capacity_ah: 34.0,
                r_internal: 0.1,
                aux_power: 500.0,
            },
            drag_area: Vec3::new(1.2, 1.2, 3.5),
            gear_height: 0.6,
            crash_speed: 2.5,
            max_rate: 1.2,
        }
    }

    /// Generic two-seat eVTOL (16 rotors on 8 coaxial arms, 620 kg, ~25 kWh).
    pub fn evtol_two_seat() -> Self {
        Airframe {
            vehicle: VehicleParams::evtol_two_seat(),
            omega_max: 230.0, // ~2200 rpm
            pitch_length: 0.16,
            prop_radius: 0.85,
            motor_tau_up: 0.2,
            motor_tau_down: 0.25,
            motor_efficiency: 0.85,
            battery: BatterySpec {
                cells: 104,
                capacity_ah: 62.0,
                r_internal: 0.06,
                aux_power: 1000.0,
            },
            drag_area: Vec3::new(2.5, 2.5, 8.0),
            gear_height: 0.8,
            crash_speed: 2.0,
            max_rate: 0.8,
        }
    }

    /// Builds an airframe from one of gym-pybullet-drones' URDF files.
    ///
    /// The URDF gives `kf`/`km` per RPM^2; they are converted to rad/s.
    pub fn from_urdf(model: DroneModel) -> Result<Self, UrdfError> {
        let urdf = model.urdf();
        let p = |name| get_urdf_parameter(urdf, name).map(|v| v as f32);
        let mass = p(UrdfParam::M)?;
        let inertia = Vec3::new(p(UrdfParam::Ixx)?, p(UrdfParam::Iyy)?, p(UrdfParam::Izz)?);
        let rpm_to_rad = 60.0 / (2.0 * core::f32::consts::PI);
        let kf = p(UrdfParam::Kf)? * rpm_to_rad * rpm_to_rad;
        let km = p(UrdfParam::Km)? * rpm_to_rad * rpm_to_rad;
        let thrust2weight = p(UrdfParam::Thrust2Weight)?;
        let max_thrust = thrust2weight * mass * G / 4.0;

        // BaseAviary: z_torque = -t0 + t1 - t2 + t3, negated for the racer.
        let spins: [f32; 4] = match model {
            DroneModel::Race => [-1.0, 1.0, -1.0, 1.0],
            _ => [1.0, -1.0, 1.0, -1.0],
        };
        let positions = rotor_positions(urdf)?;
        let rotors: Vec<_> = positions
            .iter()
            .zip(spins)
            .map(|(&(x, y, z), spin)| dronelab_core::Rotor {
                position: Vec3::new(x as f32, y as f32, z as f32),
                spin,
                torque_ratio: km / kf,
                max_thrust,
            })
            .collect();

        Ok(Airframe {
            vehicle: VehicleParams::new(model.value(), mass, inertia, &rotors),
            omega_max: libm::sqrtf(max_thrust / kf),
            pitch_length: 0.0035,
            prop_radius: p(UrdfParam::PropRadius)?,
            motor_tau_up: 0.03,
            motor_tau_down: 0.05,
            motor_efficiency: 0.8,
            battery: BatterySpec {
                cells: 1,
                capacity_ah: 0.25,
                r_internal: 0.15,
                aux_power: 0.3,
            },
            drag_area: Vec3::new(0.0015, 0.0015, 0.003),
            gear_height: 0.015,
            crash_speed: 3.0,
            max_rate: 8.0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn urdf_crazyflie_matches_core_preset() {
        let a = Airframe::crazyflie_x();
        let core = VehicleParams::crazyflie_x();
        assert_eq!(a.vehicle.mass, core.mass);
        assert_eq!(a.vehicle.inertia, core.inertia);
        for (u, c) in a.vehicle.rotors().iter().zip(core.rotors()) {
            assert!((u.position - c.position).norm() < 1e-6);
            assert_eq!(u.spin, c.spin);
            assert!((u.torque_ratio - c.torque_ratio).abs() < 1e-6);
            assert!((u.max_thrust - c.max_thrust).abs() < 1e-6);
        }
        // About 21,700 RPM at full thrust, as in gym-pybullet-drones.
        let rpm = a.omega_max * 60.0 / (2.0 * core::f32::consts::PI);
        assert!((rpm - 21_700.0).abs() < 300.0, "{rpm}");
    }

    #[test]
    fn evtol_hover_power_is_plausible() {
        // Shaft power km * w^3 summed over rotors at hover, in kW.
        for (a, lo, hi) in [
            (Airframe::evtol_single_seat(), 15.0, 45.0),
            (Airframe::evtol_two_seat(), 50.0, 130.0),
        ] {
            let hover = a.hover_fraction();
            let w = a.omega_max * libm::sqrtf(hover);
            let p: f32 = (0..a.vehicle.rotor_count())
                .map(|i| a.kf(i) * a.vehicle.rotors()[i].torque_ratio * w * w * w)
                .sum::<f32>()
                / 1000.0;
            assert!(p > lo && p < hi, "{}: {p} kW", a.vehicle.name);
        }
    }
}

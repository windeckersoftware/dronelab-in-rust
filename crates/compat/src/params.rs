//! Drone constants, as `BaseAviary._parseURDFParameters` and `__init__`
//! compute them.

use dronelab::DroneModel;
use dronelab::urdf::{UrdfError, UrdfParam, get_urdf_parameter, rotor_positions};

use crate::math::{Mat3, Vec3};

/// Gravity upstream uses everywhere (`BaseAviary.G`, the controllers' `g`).
pub const G: f64 = 9.8;

#[derive(Debug, Clone, PartialEq)]
pub struct Params {
    pub model: DroneModel,
    pub m: f64,
    /// Arm length `L`.
    pub l: f64,
    pub thrust2weight: f64,
    pub j: Mat3,
    pub j_inv: Mat3,
    pub kf: f64,
    pub km: f64,
    pub collision_h: f64,
    pub collision_r: f64,
    pub collision_z_offset: f64,
    pub max_speed_kmh: f64,
    pub gnd_eff_coeff: f64,
    pub prop_radius: f64,
    pub drag_coeff: Vec3,
    pub dw_coeff_1: f64,
    pub dw_coeff_2: f64,
    pub dw_coeff_3: f64,
    pub gravity: f64,
    pub hover_rpm: f64,
    pub max_rpm: f64,
    pub max_thrust: f64,
    pub max_xy_torque: f64,
    pub max_z_torque: f64,
    pub gnd_eff_h_clip: f64,
    /// Centre of mass of each propeller link, body frame (where PyBullet
    /// applies the rotor forces).
    pub prop_pos: [Vec3; 4],
}

impl Params {
    pub fn load(model: DroneModel) -> Result<Self, UrdfError> {
        let urdf = model.urdf();
        let p = |name| get_urdf_parameter(urdf, name);
        let (ixx, iyy, izz) = (p(UrdfParam::Ixx)?, p(UrdfParam::Iyy)?, p(UrdfParam::Izz)?);
        let m = p(UrdfParam::M)?;
        let l = p(UrdfParam::Arm)?;
        let kf = p(UrdfParam::Kf)?;
        let km = p(UrdfParam::Km)?;
        let thrust2weight = p(UrdfParam::Thrust2Weight)?;
        let gravity = G * m;
        let hover_rpm = (gravity / (4.0 * kf)).sqrt();
        let max_rpm = ((thrust2weight * gravity) / (4.0 * kf)).sqrt();
        let max_thrust = 4.0 * kf * max_rpm.powi(2);
        let max_xy_torque = match model {
            DroneModel::Cf2p => l * kf * max_rpm.powi(2),
            DroneModel::Cf2x | DroneModel::Race => (2.0 * l * kf * max_rpm.powi(2)) / 2f64.sqrt(),
        };
        let gnd_eff_coeff = p(UrdfParam::GndEffCoeff)?;
        let prop_radius = p(UrdfParam::PropRadius)?;
        let props = rotor_positions(urdf)?;
        let mut prop_pos = [[0.0; 3]; 4];
        for (dst, &(x, y, z)) in prop_pos.iter_mut().zip(&props) {
            *dst = [x, y, z];
        }
        let drag_xy = p(UrdfParam::DragCoeffXy)?;
        Ok(Params {
            model,
            m,
            l,
            thrust2weight,
            j: [[ixx, 0.0, 0.0], [0.0, iyy, 0.0], [0.0, 0.0, izz]],
            j_inv: [
                [1.0 / ixx, 0.0, 0.0],
                [0.0, 1.0 / iyy, 0.0],
                [0.0, 0.0, 1.0 / izz],
            ],
            kf,
            km,
            collision_h: p(UrdfParam::Length)?,
            collision_r: p(UrdfParam::Radius)?,
            collision_z_offset: p(UrdfParam::CollisionZOffset)?,
            max_speed_kmh: p(UrdfParam::MaxSpeedKmh)?,
            gnd_eff_coeff,
            prop_radius,
            drag_coeff: [drag_xy, drag_xy, p(UrdfParam::DragCoeffZ)?],
            dw_coeff_1: p(UrdfParam::DwCoeff1)?,
            dw_coeff_2: p(UrdfParam::DwCoeff2)?,
            dw_coeff_3: p(UrdfParam::DwCoeff3)?,
            gravity,
            hover_rpm,
            max_rpm,
            max_thrust,
            max_xy_torque,
            max_z_torque: 2.0 * km * max_rpm.powi(2),
            gnd_eff_h_clip: 0.25
                * prop_radius
                * ((15.0 * max_rpm.powi(2) * kf * gnd_eff_coeff) / max_thrust).sqrt(),
            prop_pos,
        })
    }

    /// `BaseAviary`'s default start: drones in a diagonal line, 10 cm up.
    pub fn default_init_xyzs(&self, num_drones: usize) -> Vec<Vec3> {
        let z = self.collision_h / 2.0 - self.collision_z_offset + 0.1;
        (0..num_drones)
            .map(|i| [i as f64 * 4.0 * self.l, i as f64 * 4.0 * self.l, z])
            .collect()
    }
}

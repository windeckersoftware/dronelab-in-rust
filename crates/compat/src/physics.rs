//! `BaseAviary`'s physics: the explicit `DYN` model, the forces upstream
//! hands to PyBullet in the `PYB*` modes (rotor thrust and torque, ground
//! effect, drag, downwash), and a stand-in for PyBullet's rigid-body step.

use crate::math::*;
use crate::params::{G, Params};
use dronelab::DroneModel;

/// `Physics` enum values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Physics {
    Pyb,
    Dyn,
    PybGnd,
    PybDrag,
    PybDw,
    PybGndDragDw,
}

impl Physics {
    pub fn value(self) -> &'static str {
        match self {
            Physics::Pyb => "pyb",
            Physics::Dyn => "dyn",
            Physics::PybGnd => "pyb_gnd",
            Physics::PybDrag => "pyb_drag",
            Physics::PybDw => "pyb_dw",
            Physics::PybGndDragDw => "pyb_gnd_drag_dw",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        [
            Physics::Pyb,
            Physics::Dyn,
            Physics::PybGnd,
            Physics::PybDrag,
            Physics::PybDw,
            Physics::PybGndDragDw,
        ]
        .into_iter()
        .find(|p| p.value() == s)
    }

    fn ground(self) -> bool {
        matches!(self, Physics::PybGnd | Physics::PybGndDragDw)
    }
    fn drag(self) -> bool {
        matches!(self, Physics::PybDrag | Physics::PybGndDragDw)
    }
    fn downwash(self) -> bool {
        matches!(self, Physics::PybDw | Physics::PybGndDragDw)
    }
}

/// One drone in the "engine": what PyBullet would hold.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Body {
    pub pos: Vec3,
    pub quat: QuatXyzw,
    /// World frame.
    pub vel: Vec3,
    /// World frame, as `getBaseVelocity` returns it.
    pub ang_vel: Vec3,
    /// External force and torque accumulated for the next step, world frame.
    pub force: Vec3,
    pub torque: Vec3,
    /// Orientation PyBullet's cached link transforms hold: the one from
    /// before the latest step, unless forward kinematics ran since.
    /// Link-frame forces are rotated with it.
    pub link_quat: QuatXyzw,
}

impl Body {
    pub fn new(pos: Vec3, quat: QuatXyzw) -> Self {
        Body {
            pos,
            quat,
            vel: [0.0; 3],
            ang_vel: [0.0; 3],
            force: [0.0; 3],
            torque: [0.0; 3],
            link_quat: quat,
        }
    }
}

/// An external force or torque as passed to `p.applyExternalForce` /
/// `p.applyExternalTorque` (link frame), kept for testing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Applied {
    Force {
        link: usize,
        f: Vec3,
    },
    Torque {
        link: usize,
        t: Vec3,
    },
    /// Not a force: `p.getLinkStates(..., computeForwardKinematics=1)`,
    /// which refreshes the cached link transforms that later link-frame
    /// forces in the same substep are rotated with.
    ForwardKinematics,
}

/// Index of `center_of_mass_link` in the drone URDFs.
pub const COM_LINK: usize = 4;

/// PyBullet's default multibody damping (left on by upstream).
pub const PYBULLET_DAMPING: f64 = 0.04;

/// Applies a link-frame force/torque to a body, as PyBullet does: rotor
/// links sit at the propellers' centres of mass.
pub fn apply(p: &Params, body: &mut Body, a: Applied) {
    let r = pb_matrix(body.link_quat);
    match a {
        Applied::Force { link, f } => {
            let fw = mat_vec(&r, f);
            body.force = add(body.force, fw);
            if link < 4 {
                // The force vector comes from the cached (lagged) link
                // transform; the lever arm from the current pose.
                let arm = mat_vec(&pb_matrix(body.quat), p.prop_pos[link]);
                body.torque = add(body.torque, cross(arm, fw));
            }
        }
        Applied::Torque { t, .. } => body.torque = add(body.torque, mat_vec(&r, t)),
        Applied::ForwardKinematics => body.link_quat = body.quat,
    }
}

/// `_physics`: rotor thrusts on the propeller links, yaw torque on the
/// centre-of-mass link.
pub fn rotor_forces(p: &Params, rpm: &[f64; 4], out: &mut Vec<Applied>) {
    let forces = rpm.map(|r| r * r * p.kf);
    let mut torques = rpm.map(|r| r * r * p.km);
    if p.model == DroneModel::Race {
        torques = torques.map(|t| -t);
    }
    let z_torque = -torques[0] + torques[1] - torques[2] + torques[3];
    for (i, f) in forces.iter().enumerate() {
        out.push(Applied::Force {
            link: i,
            f: [0.0, 0.0, *f],
        });
    }
    out.push(Applied::Torque {
        link: COM_LINK,
        t: [0.0, 0.0, z_torque],
    });
}

/// `_groundEffect`, from the propellers' heights in the engine state and
/// the stored roll/pitch.
pub fn ground_effect(p: &Params, rpm: &[f64; 4], body: &Body, rpy: Vec3, out: &mut Vec<Applied>) {
    out.push(Applied::ForwardKinematics);
    let r = pb_matrix(body.quat);
    let mut effects = [0.0; 4];
    for (i, e) in effects.iter_mut().enumerate() {
        let h = body.pos[2] + mat_vec(&r, p.prop_pos[i])[2];
        let h = if h < p.gnd_eff_h_clip {
            p.gnd_eff_h_clip
        } else {
            h
        };
        *e = rpm[i] * rpm[i] * p.kf * p.gnd_eff_coeff * (p.prop_radius / (4.0 * h)).powi(2);
    }
    if rpy[0].abs() < std::f64::consts::FRAC_PI_2 && rpy[1].abs() < std::f64::consts::FRAC_PI_2 {
        for (i, e) in effects.iter().enumerate() {
            out.push(Applied::Force {
                link: i,
                f: [0.0, 0.0, *e],
            });
        }
    }
}

/// `_drag`: from the stored attitude and velocity and the previous action.
/// Upstream rotates `drag_factors * vel` into the body frame and applies it
/// in the link frame, so the force ends up along the world axes.
pub fn drag(p: &Params, rpm: &[f64; 4], quat: QuatXyzw, vel: Vec3, out: &mut Vec<Applied>) {
    let base_rot = pb_matrix(quat);
    let s: f64 = rpm
        .iter()
        .map(|r| 2.0 * std::f64::consts::PI * r / 60.0)
        .sum();
    let factors = scale(p.drag_coeff, -s);
    let f = mat_vec(&transpose(&base_rot), mul(factors, vel));
    out.push(Applied::Force { link: COM_LINK, f });
}

/// `_downwash` on drone `n` from every drone above it.
pub fn downwash(p: &Params, positions: &[Vec3], n: usize, out: &mut Vec<Applied>) {
    for other in positions {
        let delta_z = other[2] - positions[n][2];
        let delta_xy =
            ((other[0] - positions[n][0]).powi(2) + (other[1] - positions[n][1]).powi(2)).sqrt();
        if delta_z > 0.0 && delta_xy < 10.0 {
            let alpha = p.dw_coeff_1 * (p.prop_radius / (4.0 * delta_z)).powi(2);
            let beta = p.dw_coeff_2 * delta_z + p.dw_coeff_3;
            out.push(Applied::Force {
                link: COM_LINK,
                f: [0.0, 0.0, -alpha * (-0.5 * (delta_xy / beta).powi(2)).exp()],
            });
        }
    }
}

/// All the forces upstream applies to drone `n` in one physics substep.
#[allow(clippy::too_many_arguments)]
pub fn substep_forces(
    p: &Params,
    physics: Physics,
    rpm: &[f64; 4],
    last_rpm: &[f64; 4],
    body: &Body,
    stored_quat: QuatXyzw,
    stored_rpy: Vec3,
    stored_vel: Vec3,
    stored_positions: &[Vec3],
    n: usize,
    out: &mut Vec<Applied>,
) {
    rotor_forces(p, rpm, out);
    if physics.ground() {
        ground_effect(p, rpm, body, stored_rpy, out);
    }
    if physics.drag() {
        drag(p, last_rpm, stored_quat, stored_vel, out);
    }
    if physics.downwash() {
        downwash(p, stored_positions, n, out);
    }
}

/// `_integrateQ`.
pub fn integrate_q(quat: QuatXyzw, omega: Vec3, dt: f64) -> QuatXyzw {
    let omega_norm = norm(omega);
    let [p, q, r] = omega;
    // np.isclose(omega_norm, 0): |a - b| <= atol + rtol * |b| = 1e-8.
    if omega_norm.abs() <= 1e-8 {
        return quat;
    }
    let lambda = [
        [0.0, r, -q, p],
        [-r, 0.0, p, q],
        [q, -p, 0.0, r],
        [-p, -q, -r, 0.0],
    ]
    .map(|row| row.map(|v| v * 0.5));
    let theta = omega_norm * dt / 2.0;
    let (s, c) = theta.sin_cos();
    let mut out = [0.0; 4];
    for (i, o) in out.iter_mut().enumerate() {
        let mut acc = 0.0;
        for j in 0..4 {
            let eye = if i == j { c } else { 0.0 };
            acc += (eye + 2.0 / omega_norm * lambda[i][j] * s) * quat[j];
        }
        *o = acc;
    }
    out
}

/// `_dynamics`: explicit Euler on the stored state; returns the new
/// engine body and body rates. The engine's angular velocity is the
/// *pre-step* rotation times the new rates, as upstream stores it.
pub fn dyn_step(p: &Params, rpm: &[f64; 4], body: &Body, rpy_rates: Vec3, dt: f64) -> (Body, Vec3) {
    let rotation = pb_matrix(body.quat);
    let forces = rpm.map(|r| r * r * p.kf);
    let thrust = [0.0, 0.0, forces.iter().sum()];
    let force_world = sub(mat_vec(&rotation, thrust), [0.0, 0.0, p.gravity]);
    let mut z_torques = rpm.map(|r| r * r * p.km);
    if p.model == DroneModel::Race {
        z_torques = z_torques.map(|t| -t);
    }
    let z_torque = -z_torques[0] + z_torques[1] - z_torques[2] + z_torques[3];
    let (x_torque, y_torque) = match p.model {
        DroneModel::Race => (
            (forces[0] + forces[1] - forces[2] - forces[3]) * (p.l / 2f64.sqrt()),
            (-forces[0] + forces[1] + forces[2] - forces[3]) * (p.l / 2f64.sqrt()),
        ),
        DroneModel::Cf2x => (
            -(forces[0] + forces[1] - forces[2] - forces[3]) * (p.l / 2f64.sqrt()),
            (-forces[0] + forces[1] + forces[2] - forces[3]) * (p.l / 2f64.sqrt()),
        ),
        DroneModel::Cf2p => (
            (forces[1] - forces[3]) * p.l,
            (-forces[0] + forces[2]) * p.l,
        ),
    };
    let torques = sub(
        [x_torque, y_torque, z_torque],
        cross(rpy_rates, mat_vec(&p.j, rpy_rates)),
    );
    let rpy_rates_deriv = mat_vec(&p.j_inv, torques);
    let acc = scale(force_world, 1.0 / p.m);
    let vel = add(body.vel, scale(acc, dt));
    let rates = add(rpy_rates, scale(rpy_rates_deriv, dt));
    let pos = add(body.pos, scale(vel, dt));
    let quat = integrate_q(body.quat, rates, dt);
    let next = Body {
        pos,
        quat,
        vel,
        ang_vel: mat_vec(&rotation, rates),
        force: [0.0; 3],
        torque: [0.0; 3],
        link_quat: quat,
    };
    (next, rates)
}

/// Stand-in for `p.stepSimulation()` on a drone with no contacts: Bullet
/// multibody dynamics (gravity, external forces, default damping,
/// gyroscopic term) with semi-implicit Euler and an exponential-map
/// orientation update. Not bit-identical to Bullet; see the README for
/// the measured agreement. Clears the accumulated forces.
pub fn pyb_step(p: &Params, body: &mut Body, dt: f64) {
    let r = pb_matrix(body.quat);
    let rt = transpose(&r);
    let i = [p.j[0][0], p.j[1][1], p.j[2][2]];
    let v_b = mat_vec(&rt, body.vel);
    let w_b = mat_vec(&rt, body.ang_vel);
    let d = PYBULLET_DAMPING;

    let f_b = mat_vec(&rt, add(body.force, [0.0, 0.0, -G * p.m]));
    let f_b = sub(f_b, scale(v_b, p.m * (d + d * norm(v_b))));
    let tau_b = mat_vec(&rt, body.torque);
    let tau_b = sub(tau_b, scale(mul(i, w_b), d + d * norm(w_b)));
    let tau_b = sub(tau_b, cross(w_b, mul(i, w_b)));
    let w_dot = [tau_b[0] / i[0], tau_b[1] / i[1], tau_b[2] / i[2]];

    // Bullet keeps base velocities in the world frame: accelerations found
    // in the base frame are rotated back with the pre-step orientation.
    body.vel = add(body.vel, scale(mat_vec(&r, f_b), dt / p.m));
    body.ang_vel = add(body.ang_vel, scale(mat_vec(&r, w_dot), dt));
    body.pos = add(body.pos, scale(body.vel, dt));
    body.link_quat = body.quat;
    body.quat = quat_exp_step(body.quat, mat_vec(&rt, body.ang_vel), dt);
    body.force = [0.0; 3];
    body.torque = [0.0; 3];
}

/// q <- q * exp(w_b dt / 2), normalised (Bullet's `pQuatUpdateFun`).
fn quat_exp_step(q: QuatXyzw, w_b: Vec3, dt: f64) -> QuatXyzw {
    let angle = norm(w_b);
    let half = 0.5 * angle * dt;
    let k = if angle < 0.001 {
        0.5 * dt - dt * dt * dt * 0.020833333333 * angle * angle
    } else {
        half.sin() / angle
    };
    let dq = [w_b[0] * k, w_b[1] * k, w_b[2] * k, half.cos()];
    let [x1, y1, z1, w1] = q;
    let [x2, y2, z2, w2] = dq;
    let out = [
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ];
    let n = (out[0] * out[0] + out[1] * out[1] + out[2] * out[2] + out[3] * out[3]).sqrt();
    out.map(|v| v / n)
}

/// Approximate contact with PyBullet's ground plane for the drone's
/// collision cylinder: removes penetration and downward speed, with
/// friction. PyBullet's constraint solver is not reproduced.
pub fn ground_contact(p: &Params, body: &mut Body, dt: f64) -> bool {
    let r = pb_matrix(body.quat);
    let axis = col(&r, 2);
    let center = add(body.pos, mat_vec(&r, [0.0, 0.0, p.collision_z_offset]));
    let az = axis[2].abs().min(1.0);
    let lowest = center[2] - 0.5 * p.collision_h * az - p.collision_r * (1.0 - az * az).sqrt();
    if lowest >= 0.0 {
        return false;
    }
    body.pos[2] -= lowest;
    if body.vel[2] < 0.0 {
        body.vel[2] = 0.0;
    }
    let k = (1.0 - 10.0 * dt).max(0.0);
    body.vel[0] *= k;
    body.vel[1] *= k;
    body.ang_vel = scale(body.ang_vel, k);
    true
}

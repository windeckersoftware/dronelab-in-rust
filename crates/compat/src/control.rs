//! Ports of upstream's controllers: `DSLPIDControl`, `CTBRControl` and
//! `MRACControl`, with their `BaseControl` interface.

use dronelab::DroneModel;

use crate::math::*;
use crate::mrac_tables;
use crate::params::Params;

/// Output of `computeControl`: RPM per motor, position error, and yaw
/// error (MRAC returns a full roll/pitch/yaw error instead).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ControlOutput {
    pub rpm: [f64; 4],
    pub pos_e: Vec3,
    pub yaw_e: f64,
    /// MRAC's `rpy_e`; for DSL this is `[0, 0, yaw_e]`.
    pub rpy_e: Vec3,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Target {
    pub pos: Vec3,
    pub rpy: Vec3,
    pub vel: Vec3,
    pub rpy_rates: Vec3,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ControlError {
    UnsupportedModel(&'static str),
    /// `setPIDCoefficients` on a controller without PID gains.
    NotPid,
}

impl std::fmt::Display for ControlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ControlError::UnsupportedModel(m) => write!(f, "{m}"),
            ControlError::NotPid => f.write_str(
                "in BaseControl.setPIDCoefficients(), not all PID coefficients exist as attributes in the instantiated control class.",
            ),
        }
    }
}

impl std::error::Error for ControlError {}

/// The six PID gain vectors (`P_COEFF_FOR` ... `D_COEFF_TOR`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PidCoefficients {
    pub p_coeff_for: Vec3,
    pub i_coeff_for: Vec3,
    pub d_coeff_for: Vec3,
    pub p_coeff_tor: Vec3,
    pub i_coeff_tor: Vec3,
    pub d_coeff_tor: Vec3,
}

/// Arguments of `setPIDCoefficients`; `None` keeps the current value.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct SetPid {
    pub p_coeff_pos: Option<Vec3>,
    pub i_coeff_pos: Option<Vec3>,
    pub d_coeff_pos: Option<Vec3>,
    pub p_coeff_att: Option<Vec3>,
    pub i_coeff_att: Option<Vec3>,
    pub d_coeff_att: Option<Vec3>,
}

const PWM2RPM_SCALE: f64 = 0.2685;
const PWM2RPM_CONST: f64 = 4070.3;
const MIN_PWM: f64 = 20000.0;
const MAX_PWM: f64 = 65535.0;
const MIXER_X: [[f64; 3]; 4] = [
    [-0.5, -0.5, -1.0],
    [-0.5, 0.5, 1.0],
    [0.5, 0.5, -1.0],
    [0.5, -0.5, 1.0],
];
const MIXER_P: [[f64; 3]; 4] = [
    [0.0, -1.0, -1.0],
    [1.0, 0.0, 1.0],
    [0.0, 1.0, -1.0],
    [-1.0, 0.0, 1.0],
];

/// `DSLPIDControl`: cascaded position and attitude PID for Crazyflies,
/// returning RPMs through the Crazyflie PWM map.
#[derive(Debug, Clone)]
pub struct DslPidControl {
    pub model: DroneModel,
    /// `BaseControl.GRAVITY` (M*g) and `KF`.
    pub gravity: f64,
    pub kf: f64,
    pub km: f64,
    pub control_counter: u64,
    pub coeffs: PidCoefficients,
    mixer: [[f64; 3]; 4],
    last_rpy: Vec3,
    integral_pos_e: Vec3,
    integral_rpy_e: Vec3,
}

impl DslPidControl {
    pub fn new(model: DroneModel, g: f64) -> Result<Self, ControlError> {
        let mixer = match model {
            DroneModel::Cf2x => MIXER_X,
            DroneModel::Cf2p => MIXER_P,
            DroneModel::Race => {
                return Err(ControlError::UnsupportedModel(
                    "in DSLPIDControl.__init__(), DSLPIDControl requires DroneModel.CF2X or DroneModel.CF2P",
                ));
            }
        };
        let p = Params::load(model).expect("bundled URDF is valid");
        Ok(DslPidControl {
            model,
            gravity: g * p.m,
            kf: p.kf,
            km: p.km,
            control_counter: 0,
            coeffs: PidCoefficients {
                p_coeff_for: [0.4, 0.4, 1.25],
                i_coeff_for: [0.05, 0.05, 0.05],
                d_coeff_for: [0.2, 0.2, 0.5],
                p_coeff_tor: [70000.0, 70000.0, 60000.0],
                i_coeff_tor: [0.0, 0.0, 500.0],
                d_coeff_tor: [20000.0, 20000.0, 12000.0],
            },
            mixer,
            last_rpy: [0.0; 3],
            integral_pos_e: [0.0; 3],
            integral_rpy_e: [0.0; 3],
        })
    }

    pub fn reset(&mut self) {
        self.control_counter = 0;
        self.last_rpy = [0.0; 3];
        self.integral_pos_e = [0.0; 3];
        self.integral_rpy_e = [0.0; 3];
    }

    pub fn set_pid_coefficients(&mut self, s: &SetPid) {
        let c = &mut self.coeffs;
        let set = |dst: &mut Vec3, v: Option<Vec3>| {
            if let Some(v) = v {
                *dst = v;
            }
        };
        set(&mut c.p_coeff_for, s.p_coeff_pos);
        set(&mut c.i_coeff_for, s.i_coeff_pos);
        set(&mut c.d_coeff_for, s.d_coeff_pos);
        set(&mut c.p_coeff_tor, s.p_coeff_att);
        set(&mut c.i_coeff_tor, s.i_coeff_att);
        set(&mut c.d_coeff_tor, s.d_coeff_att);
    }

    /// `computeControl`. `cur_ang_vel` is unused, as upstream.
    pub fn compute_control(
        &mut self,
        dt: f64,
        cur_pos: Vec3,
        cur_quat: QuatXyzw,
        cur_vel: Vec3,
        _cur_ang_vel: Vec3,
        t: &Target,
    ) -> ControlOutput {
        self.control_counter += 1;
        let (thrust, target_euler, pos_e) =
            self.position_control(dt, cur_pos, cur_quat, cur_vel, t.pos, t.rpy, t.vel);
        let rpm = self.attitude_control(dt, thrust, cur_quat, target_euler, t.rpy_rates);
        let cur_rpy = pb_euler(cur_quat);
        let yaw_e = target_euler[2] - cur_rpy[2];
        ControlOutput {
            rpm,
            pos_e,
            yaw_e,
            rpy_e: [0.0, 0.0, yaw_e],
        }
    }

    /// `computeControlFromState` with the 20-value state vector.
    pub fn compute_control_from_state(
        &mut self,
        dt: f64,
        state: &[f64; 20],
        t: &Target,
    ) -> ControlOutput {
        let s = |i: usize| [state[i], state[i + 1], state[i + 2]];
        self.compute_control(
            dt,
            s(0),
            [state[3], state[4], state[5], state[6]],
            s(10),
            s(13),
            t,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn position_control(
        &mut self,
        dt: f64,
        cur_pos: Vec3,
        cur_quat: QuatXyzw,
        cur_vel: Vec3,
        target_pos: Vec3,
        target_rpy: Vec3,
        target_vel: Vec3,
    ) -> (f64, Vec3, Vec3) {
        let c = &self.coeffs;
        let cur_rotation = pb_matrix(cur_quat);
        let pos_e = sub(target_pos, cur_pos);
        let vel_e = sub(target_vel, cur_vel);
        let ip = add(self.integral_pos_e, scale(pos_e, dt));
        let mut ip = [
            clip(ip[0], -2.0, 2.0),
            clip(ip[1], -2.0, 2.0),
            clip(ip[2], -2.0, 2.0),
        ];
        ip[2] = clip(ip[2], -0.15, 0.15);
        self.integral_pos_e = ip;
        let target_thrust = add(
            add(
                add(mul(c.p_coeff_for, pos_e), mul(c.i_coeff_for, ip)),
                mul(c.d_coeff_for, vel_e),
            ),
            [0.0, 0.0, self.gravity],
        );
        let scalar_thrust = dot(target_thrust, col(&cur_rotation, 2)).max(0.0);
        let thrust = ((scalar_thrust / (4.0 * self.kf)).sqrt() - PWM2RPM_CONST) / PWM2RPM_SCALE;
        let target_z_ax = scale(target_thrust, 1.0 / norm(target_thrust));
        let target_x_c = [target_rpy[2].cos(), target_rpy[2].sin(), 0.0];
        let yz = cross(target_z_ax, target_x_c);
        let target_y_ax = scale(yz, 1.0 / norm(yz));
        let target_x_ax = cross(target_y_ax, target_z_ax);
        let target_rotation = from_cols(target_x_ax, target_y_ax, target_z_ax);
        let target_euler = scipy_matrix_to_euler_xyz(&target_rotation);
        (thrust, target_euler, pos_e)
    }

    fn attitude_control(
        &mut self,
        dt: f64,
        thrust: f64,
        cur_quat: QuatXyzw,
        target_euler: Vec3,
        target_rpy_rates: Vec3,
    ) -> [f64; 4] {
        let c = &self.coeffs;
        let cur_rotation = pb_matrix(cur_quat);
        let cur_rpy = pb_euler(cur_quat);
        // Upstream unpacks the scalar-last quaternion as `w, x, y, z` and
        // packs it back in the same order, so the round trip is a no-op.
        let target_rotation = scipy_quat_to_matrix(scipy_euler_xyz_to_quat(target_euler));
        let a = mat_mul(&transpose(&target_rotation), &cur_rotation);
        let b = mat_mul(&transpose(&cur_rotation), &target_rotation);
        let rot_e = [a[2][1] - b[2][1], a[0][2] - b[0][2], a[1][0] - b[1][0]];
        let rpy_rates_e = sub(
            target_rpy_rates,
            scale(sub(cur_rpy, self.last_rpy), 1.0 / dt),
        );
        self.last_rpy = cur_rpy;
        let ir = sub(self.integral_rpy_e, scale(rot_e, dt));
        let mut ir = [
            clip(ir[0], -1500.0, 1500.0),
            clip(ir[1], -1500.0, 1500.0),
            clip(ir[2], -1500.0, 1500.0),
        ];
        ir[0] = clip(ir[0], -1.0, 1.0);
        ir[1] = clip(ir[1], -1.0, 1.0);
        self.integral_rpy_e = ir;
        let tt = add(
            add(
                scale(mul(c.p_coeff_tor, rot_e), -1.0),
                mul(c.d_coeff_tor, rpy_rates_e),
            ),
            mul(c.i_coeff_tor, ir),
        );
        let tt = [
            clip(tt[0], -3200.0, 3200.0),
            clip(tt[1], -3200.0, 3200.0),
            clip(tt[2], -3200.0, 3200.0),
        ];
        let mut rpm = [0.0; 4];
        for (r, row) in rpm.iter_mut().zip(&self.mixer) {
            let pwm = clip(thrust + dot(*row, tt), MIN_PWM, MAX_PWM);
            *r = PWM2RPM_SCALE * pwm + PWM2RPM_CONST;
        }
        rpm
    }
}

/// `CTBRControl`: collective thrust and body rates, for Betaflight SITL.
/// Quaternions are `[w, x, y, z]` here, as upstream.
#[derive(Debug, Clone)]
pub struct CtbrControl {
    pub model: DroneModel,
    pub gravity: f64,
    pub kf: f64,
    pub km: f64,
    pub control_counter: u64,
}

impl CtbrControl {
    pub fn new(model: DroneModel, g: f64) -> Self {
        let p = Params::load(model).expect("bundled URDF is valid");
        CtbrControl {
            model,
            gravity: g * p.m,
            kf: p.kf,
            km: p.km,
            control_counter: 0,
        }
    }

    pub fn reset(&mut self) {
        self.control_counter = 0;
    }

    /// Returns `(normalised thrust, roll rate, pitch rate, yaw rate)`.
    pub fn compute_control(
        &mut self,
        cur_pos: Vec3,
        cur_quat_wxyz: QuatWxyz,
        cur_vel: Vec3,
        target_pos: Vec3,
        target_vel: Vec3,
    ) -> [f64; 4] {
        let g = [0.0, 0.0, -9.8];
        let k_p = [3.0, 3.0, 8.0];
        let k_d = [2.5, 2.5, 5.0];
        let k_rates = [5.0, 5.0, 1.0];
        let tar_acc = sub(
            add(
                mul(k_p, sub(target_pos, cur_pos)),
                mul(k_d, sub(target_vel, cur_vel)),
            ),
            g,
        );
        let norm_thrust = dot(tar_acc, t3d_rotate_vector([0.0, 0.0, 1.0], cur_quat_wxyz));
        let z_body = t3d_normalized(tar_acc);
        let x_body = t3d_normalized(cross([0.0, 1.0, 0.0], z_body));
        let y_body = t3d_normalized(cross(z_body, x_body));
        let tar_att = t3d_mat2quat(&from_cols(x_body, y_body, z_body));
        let q_error = t3d_qmult(t3d_qconjugate(cur_quat_wxyz), tar_att);
        let mut rates = [
            2.0 * k_rates[0] * q_error[1],
            2.0 * k_rates[1] * q_error[2],
            2.0 * k_rates[2] * q_error[3],
        ];
        if q_error[0] < 0.0 {
            rates = scale(rates, -1.0);
        }
        [norm_thrust, rates[0], rates[1], rates[2]]
    }

    /// `computeControlFromState`: reorders the state's `[x, y, z, w]`.
    pub fn compute_control_from_state(
        &mut self,
        state: &[f64; 20],
        target_pos: Vec3,
        target_vel: Vec3,
    ) -> [f64; 4] {
        let s = |i: usize| [state[i], state[i + 1], state[i + 2]];
        self.compute_control(
            s(0),
            [state[6], state[3], state[4], state[5]],
            s(10),
            target_pos,
            target_vel,
        )
    }
}

/// `MRACControl`: model-reference adaptive control.
///
/// Its reference model comes from pole placement and a Lyapunov solve
/// (python-control / SciPy) at start-up. Those matrices depend only on the
/// drone, so they were computed by upstream's own code and are stored in
/// [`crate::mrac_tables`].
#[derive(Debug, Clone)]
pub struct MracControl {
    pub model: DroneModel,
    pub gravity: f64,
    pub kf: f64,
    pub control_counter: u64,
    mixer: [[f64; 3]; 4],
    /// Adaptive gains (12x4 and 4x4).
    pub kx: [[f64; 4]; 12],
    pub kr: [[f64; 4]; 4],
    xm: [f64; 12],
    tables: &'static mrac_tables::Tables,
}

impl MracControl {
    pub fn new(model: DroneModel, g: f64) -> Self {
        let p = Params::load(model).expect("bundled URDF is valid");
        let tables = mrac_tables::for_model(model);
        let mixer = if model == DroneModel::Cf2p {
            MIXER_P
        } else {
            MIXER_X
        };
        MracControl {
            model,
            gravity: g * p.m,
            kf: p.kf,
            control_counter: 0,
            mixer,
            kx: tables.kx,
            kr: tables.kr,
            xm: [0.0; 12],
            tables,
        }
    }

    /// Upstream's `reset` only zeroes the counter; the adapted gains and
    /// the reference state are kept (the reference model restarts from the
    /// next state).
    pub fn reset(&mut self) {
        self.control_counter = 0;
    }

    pub fn compute_control(
        &mut self,
        dt: f64,
        cur_pos: Vec3,
        cur_quat: QuatXyzw,
        cur_vel: Vec3,
        cur_ang_vel: Vec3,
        t: &Target,
    ) -> ControlOutput {
        let tb = self.tables;
        let cur_rpy = pb_euler(cur_quat);
        // `Rotation.from_euler('XYZ', cur_rpy).inv().apply(w)`: upstream
        // feeds PyBullet's roll/pitch/yaw to SciPy's intrinsic sequence.
        let r = scipy_euler_xyz_to_matrix(cur_rpy);
        let w_b = mat_vec(&transpose(&r), cur_ang_vel);
        let x: [f64; 12] = [
            cur_pos[0], cur_pos[1], cur_pos[2], cur_rpy[0], cur_rpy[1], cur_rpy[2], cur_vel[0],
            cur_vel[1], cur_vel[2], w_b[0], w_b[1], w_b[2],
        ];
        if self.control_counter == 0 {
            self.xm = x;
        }
        self.control_counter += 1;
        let rr: [f64; 12] = [
            t.pos[0],
            t.pos[1],
            t.pos[2],
            t.rpy[0],
            t.rpy[1],
            t.rpy[2],
            t.vel[0],
            t.vel[1],
            t.vel[2],
            t.rpy_rates[0],
            t.rpy_rates[1],
            t.rpy_rates[2],
        ];
        // rt = -Kr_ref_gain @ r   (4)
        let mut rt = [0.0; 4];
        for (i, v) in rt.iter_mut().enumerate() {
            *v = -(0..12).map(|k| tb.kr_ref_gain[i][k] * rr[k]).sum::<f64>();
        }
        // u = Kx.T @ x + Kr.T @ rt
        let mut u = [0.0; 4];
        for (i, v) in u.iter_mut().enumerate() {
            *v = (0..12).map(|k| self.kx[k][i] * x[k]).sum::<f64>()
                + (0..4).map(|k| self.kr[k][i] * rt[k]).sum::<f64>();
        }
        let mut e = [0.0; 12];
        for k in 0..12 {
            e[k] = x[k] - self.xm[k];
        }
        // eTPB = e.T @ P @ Bm   (1x4)
        let mut e_p = [0.0; 12];
        for (j, v) in e_p.iter_mut().enumerate() {
            *v = (0..12).map(|k| e[k] * tb.p[k][j]).sum::<f64>();
        }
        let mut epb = [0.0; 4];
        for (j, v) in epb.iter_mut().enumerate() {
            *v = (0..12).map(|k| e_p[k] * tb.bm[k][j]).sum::<f64>();
        }
        // Kx_dot = -Gamma_x @ x @ epb ; Kr_dot = -Gamma_r @ rt @ epb
        for k in 0..12 {
            let gx: f64 = (0..12).map(|m| tb.gamma_x[k][m] * x[m]).sum();
            for j in 0..4 {
                self.kx[k][j] += -gx * epb[j] * dt;
            }
        }
        for k in 0..4 {
            let gr: f64 = (0..4).map(|m| tb.gamma_r[k][m] * rt[m]).sum();
            for j in 0..4 {
                self.kr[k][j] += -gr * epb[j] * dt;
            }
        }
        let thrust = u[0].max(0.0);
        let tt = [
            clip(u[1], -3200.0, 3200.0),
            clip(u[2], -3200.0, 3200.0),
            clip(u[3], -3200.0, 3200.0),
        ];
        let thrust = ((thrust / (4.0 * self.kf)).sqrt() - PWM2RPM_CONST) / PWM2RPM_SCALE;
        let mut rpm = [0.0; 4];
        for (r, row) in rpm.iter_mut().zip(&self.mixer) {
            let pwm = clip(thrust + dot(*row, tt), MIN_PWM, MAX_PWM);
            *r = PWM2RPM_SCALE * pwm + PWM2RPM_CONST;
        }
        let pos_e = sub(t.pos, cur_pos);
        let rpy_e = sub(t.rpy, cur_rpy);
        // Xm += (Am @ Xm + Bm @ rt) * dt
        let mut xm_dot = [0.0; 12];
        for (k, v) in xm_dot.iter_mut().enumerate() {
            *v = (0..12).map(|m| tb.am[k][m] * self.xm[m]).sum::<f64>()
                + (0..4).map(|m| tb.bm[k][m] * rt[m]).sum::<f64>();
        }
        for k in 0..12 {
            self.xm[k] += xm_dot[k] * dt;
        }
        ControlOutput {
            rpm,
            pos_e,
            yaw_e: rpy_e[2],
            rpy_e,
        }
    }
}

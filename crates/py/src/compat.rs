//! Bindings for `dronelab-compat`, the faithful gym-pybullet-drones port.
//! The Python classes in `dronelab.compat` build on these.

use numpy::{
    PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use dronelab_compat::control::{CtbrControl, DslPidControl, MracControl, SetPid, Target};
use dronelab_compat::{
    ActionType, Aviary, AviaryBatch, AviaryConfig, DroneModel, EnvKind, G, Physics,
};

type Arr1<'py> = Bound<'py, PyArray1<f64>>;

fn err(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn model(s: &str) -> PyResult<DroneModel> {
    match s {
        "cf2x" => Ok(DroneModel::Cf2x),
        "cf2p" => Ok(DroneModel::Cf2p),
        "racer" => Ok(DroneModel::Race),
        _ => Err(err(format!("unknown drone model {s:?}"))),
    }
}

fn rows3(
    v: Option<PyReadonlyArray2<'_, f64>>,
    n: usize,
    what: &str,
) -> PyResult<Option<Vec<[f64; 3]>>> {
    let Some(a) = v else { return Ok(None) };
    if a.shape() != [n, 3] {
        return Err(err(format!(
            "invalid {what} in BaseAviary.__init__(), try {what}.reshape(NUM_DRONES,3)"
        )));
    }
    let a = a.as_array();
    Ok(Some(
        (0..n).map(|i| [a[[i, 0]], a[[i, 1]], a[[i, 2]]]).collect(),
    ))
}

#[allow(clippy::too_many_arguments)]
fn config(
    env: &str,
    act: &str,
    drone_model: &str,
    num_drones: usize,
    neighbourhood_radius: f64,
    initial_xyzs: Option<PyReadonlyArray2<'_, f64>>,
    initial_rpys: Option<PyReadonlyArray2<'_, f64>>,
    physics: &str,
    pyb_freq: u32,
    ctrl_freq: u32,
) -> PyResult<AviaryConfig> {
    let act = ActionType::parse(act).ok_or_else(|| err(format!("Unknown action type: {act}")))?;
    let kind = match env {
        "ctrl" => EnvKind::Ctrl,
        "velocity" => EnvKind::Velocity,
        "hover" => EnvKind::Hover(act),
        "multihover" => EnvKind::MultiHover(act),
        _ => return Err(err(format!("unknown env {env:?}"))),
    };
    Ok(AviaryConfig {
        kind,
        drone_model: model(drone_model)?,
        num_drones,
        neighbourhood_radius,
        initial_xyzs: rows3(initial_xyzs, num_drones, "initial_xyzs")?,
        initial_rpys: rows3(initial_rpys, num_drones, "initial_rpys")?,
        physics: Physics::parse(physics)
            .ok_or_else(|| err(format!("unknown physics {physics:?}")))?,
        pyb_freq,
        ctrl_freq,
    })
}

fn to2d<'py>(py: Python<'py>, v: Vec<f64>, rows: usize) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let cols = v.len().checked_div(rows).unwrap_or(0);
    PyArray1::from_vec(py, v).reshape([rows, cols])
}

/// One upstream aviary (BaseAviary and subclasses), stepped in Rust.
#[pyclass(module = "dronelab._native")]
pub struct CompatAviary {
    inner: Aviary,
}

#[pymethods]
impl CompatAviary {
    #[new]
    #[pyo3(signature = (env, act, drone_model, num_drones, neighbourhood_radius, initial_xyzs, initial_rpys, physics, pyb_freq, ctrl_freq))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        env: &str,
        act: &str,
        drone_model: &str,
        num_drones: usize,
        neighbourhood_radius: f64,
        initial_xyzs: Option<PyReadonlyArray2<'_, f64>>,
        initial_rpys: Option<PyReadonlyArray2<'_, f64>>,
        physics: &str,
        pyb_freq: u32,
        ctrl_freq: u32,
    ) -> PyResult<Self> {
        let cfg = config(
            env,
            act,
            drone_model,
            num_drones,
            neighbourhood_radius,
            initial_xyzs,
            initial_rpys,
            physics,
            pyb_freq,
            ctrl_freq,
        )?;
        Ok(CompatAviary {
            inner: Aviary::new(cfg).map_err(err)?,
        })
    }

    fn reset<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        to2d(py, self.inner.reset(), self.inner.num_drones())
    }

    /// Returns `(obs, reward, terminated, truncated)`.
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        action: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, f64, bool, bool)> {
        let (n, ad) = (self.inner.num_drones(), self.inner.act_dim());
        if action.shape() != [n, ad] {
            return Err(err(format!(
                "action must have shape ({n}, {ad}), got {:?}",
                action.shape()
            )));
        }
        let a: Vec<f64> = action.as_array().iter().copied().collect();
        let r = self.inner.step(&a);
        Ok((to2d(py, r.obs, n)?, r.reward, r.terminated, r.truncated))
    }

    fn state_vector<'py>(&self, py: Python<'py>, i: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if i >= self.inner.num_drones() {
            return Err(err("drone index out of range"));
        }
        Ok(PyArray1::from_slice(py, &self.inner.state_vector(i)))
    }

    fn adjacency_matrix<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        to2d(py, self.inner.adjacency_matrix(), self.inner.num_drones())
    }

    /// Stored kinematics as arrays: pos, quat, rpy, vel, ang_v, last_clipped_action.
    fn kinematics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let k = &self.inner.stored;
        let n = k.len();
        let d = PyDict::new(py);
        d.set_item("pos", to2d(py, k.iter().flat_map(|s| s.pos).collect(), n)?)?;
        d.set_item(
            "quat",
            to2d(py, k.iter().flat_map(|s| s.quat).collect(), n)?,
        )?;
        d.set_item("rpy", to2d(py, k.iter().flat_map(|s| s.rpy).collect(), n)?)?;
        d.set_item("vel", to2d(py, k.iter().flat_map(|s| s.vel).collect(), n)?)?;
        d.set_item(
            "ang_v",
            to2d(py, k.iter().flat_map(|s| s.ang_v).collect(), n)?,
        )?;
        d.set_item(
            "last_clipped_action",
            to2d(py, self.inner.last_clipped_action.concat(), n)?,
        )?;
        Ok(d)
    }

    /// Constants named as upstream's attributes.
    fn constants<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let a = &self.inner;
        let p = &a.params;
        let d = PyDict::new(py);
        for (k, v) in [
            ("G", G),
            ("M", p.m),
            ("L", p.l),
            ("THRUST2WEIGHT_RATIO", p.thrust2weight),
            ("KF", p.kf),
            ("KM", p.km),
            ("COLLISION_H", p.collision_h),
            ("COLLISION_R", p.collision_r),
            ("COLLISION_Z_OFFSET", p.collision_z_offset),
            ("MAX_SPEED_KMH", p.max_speed_kmh),
            ("GND_EFF_COEFF", p.gnd_eff_coeff),
            ("PROP_RADIUS", p.prop_radius),
            ("DW_COEFF_1", p.dw_coeff_1),
            ("DW_COEFF_2", p.dw_coeff_2),
            ("DW_COEFF_3", p.dw_coeff_3),
            ("GRAVITY", p.gravity),
            ("HOVER_RPM", p.hover_rpm),
            ("MAX_RPM", p.max_rpm),
            ("MAX_THRUST", p.max_thrust),
            ("MAX_XY_TORQUE", p.max_xy_torque),
            ("MAX_Z_TORQUE", p.max_z_torque),
            ("GND_EFF_H_CLIP", p.gnd_eff_h_clip),
            ("CTRL_TIMESTEP", a.ctrl_timestep),
            ("PYB_TIMESTEP", a.pyb_timestep),
            ("EPISODE_LEN_SEC", a.episode_len_sec),
            ("SPEED_LIMIT", a.speed_limit),
        ] {
            d.set_item(k, v)?;
        }
        d.set_item("PYB_STEPS_PER_CTRL", a.pyb_steps_per_ctrl)?;
        d.set_item("ACTION_BUFFER_SIZE", a.action_buffer_size)?;
        d.set_item("J", to2d(py, p.j.concat(), 3)?)?;
        d.set_item("J_INV", to2d(py, p.j_inv.concat(), 3)?)?;
        d.set_item("DRAG_COEFF", PyArray1::from_slice(py, &p.drag_coeff))?;
        d.set_item("INIT_XYZS", to2d(py, a.init_xyzs.concat(), a.num_drones())?)?;
        d.set_item("INIT_RPYS", to2d(py, a.init_rpys.concat(), a.num_drones())?)?;
        if !a.target_pos.is_empty() {
            let t = to2d(py, a.target_pos.concat(), a.target_pos.len())?;
            // HoverAviary keeps a single (3,) target.
            if matches!(a.cfg.kind, EnvKind::Hover(_)) {
                d.set_item("TARGET_POS", PyArray1::from_slice(py, &a.target_pos[0]))?;
            } else {
                d.set_item("TARGET_POS", t)?;
            }
        }
        Ok(d)
    }

    #[getter]
    fn step_counter(&self) -> u64 {
        self.inner.step_counter
    }
    #[getter]
    fn obs_dim(&self) -> usize {
        self.inner.obs_dim()
    }
    #[getter]
    fn act_dim(&self) -> usize {
        self.inner.act_dim()
    }
}

/// Many aviaries in parallel, with automatic reset.
#[pyclass(module = "dronelab._native")]
pub struct CompatBatch {
    inner: AviaryBatch,
    num_drones: usize,
}

#[pymethods]
impl CompatBatch {
    #[new]
    #[pyo3(signature = (num_envs, env, act, drone_model, num_drones, neighbourhood_radius, initial_xyzs, initial_rpys, physics, pyb_freq, ctrl_freq))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        num_envs: usize,
        env: &str,
        act: &str,
        drone_model: &str,
        num_drones: usize,
        neighbourhood_radius: f64,
        initial_xyzs: Option<PyReadonlyArray2<'_, f64>>,
        initial_rpys: Option<PyReadonlyArray2<'_, f64>>,
        physics: &str,
        pyb_freq: u32,
        ctrl_freq: u32,
    ) -> PyResult<Self> {
        let cfg = config(
            env,
            act,
            drone_model,
            num_drones,
            neighbourhood_radius,
            initial_xyzs,
            initial_rpys,
            physics,
            pyb_freq,
            ctrl_freq,
        )?;
        Ok(CompatBatch {
            inner: AviaryBatch::new(cfg, num_envs).map_err(err)?,
            num_drones,
        })
    }

    #[getter]
    fn num_envs(&self) -> usize {
        self.inner.num_envs()
    }

    fn reset<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let inner = &mut self.inner;
        py.detach(|| inner.reset());
        to2d(py, self.inner.obs.clone(), self.inner.num_envs())
    }

    /// `actions`: (num_envs, num_drones * act_dim). Returns
    /// `(obs, rewards, terminated, truncated, terminal_obs)`, all with one
    /// row per env (observations flattened over drones).
    #[allow(clippy::type_complexity)]
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<(
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<bool>>,
        Bound<'py, PyArray1<bool>>,
        Bound<'py, PyArray2<f64>>,
    )> {
        let (n, aw) = (self.inner.num_envs(), self.inner.act_width());
        if actions.shape() != [n, aw] {
            return Err(err(format!(
                "actions must have shape ({n}, {aw}), got {:?}",
                actions.shape()
            )));
        }
        let a: Vec<f64> = actions.as_array().iter().copied().collect();
        let inner = &mut self.inner;
        py.detach(|| inner.step(&a));
        let b = &self.inner;
        Ok((
            to2d(py, b.obs.clone(), n)?,
            PyArray1::from_slice(py, &b.rewards),
            PyArray1::from_slice(py, &b.terminated),
            PyArray1::from_slice(py, &b.truncated),
            to2d(py, b.terminal_obs.clone(), n)?,
        ))
    }

    #[getter]
    fn num_drones(&self) -> usize {
        self.num_drones
    }
}

fn v3(a: &PyReadonlyArray1<'_, f64>) -> PyResult<[f64; 3]> {
    let s = a.as_array();
    if s.len() != 3 {
        return Err(err("expected a (3,)-shaped array"));
    }
    Ok([s[0], s[1], s[2]])
}

fn v4(a: &PyReadonlyArray1<'_, f64>) -> PyResult<[f64; 4]> {
    let s = a.as_array();
    if s.len() != 4 {
        return Err(err("expected a (4,)-shaped array"));
    }
    Ok([s[0], s[1], s[2], s[3]])
}

fn opt3(a: &Option<PyReadonlyArray1<'_, f64>>) -> PyResult<Option<[f64; 3]>> {
    a.as_ref().map(v3).transpose()
}

/// `DSLPIDControl`.
#[pyclass(module = "dronelab._native")]
pub struct DslPid {
    inner: DslPidControl,
}

#[pymethods]
impl DslPid {
    #[new]
    fn new(drone_model: &str, g: f64) -> PyResult<Self> {
        Ok(DslPid {
            inner: DslPidControl::new(model(drone_model)?, g).map_err(err)?,
        })
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Returns `(rpm, pos_e, yaw_e)`.
    #[allow(clippy::too_many_arguments)]
    fn compute<'py>(
        &mut self,
        py: Python<'py>,
        dt: f64,
        cur_pos: PyReadonlyArray1<'py, f64>,
        cur_quat: PyReadonlyArray1<'py, f64>,
        cur_vel: PyReadonlyArray1<'py, f64>,
        cur_ang_vel: PyReadonlyArray1<'py, f64>,
        target_pos: PyReadonlyArray1<'py, f64>,
        target_rpy: PyReadonlyArray1<'py, f64>,
        target_vel: PyReadonlyArray1<'py, f64>,
        target_rpy_rates: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<(Arr1<'py>, Arr1<'py>, f64)> {
        let t = Target {
            pos: v3(&target_pos)?,
            rpy: v3(&target_rpy)?,
            vel: v3(&target_vel)?,
            rpy_rates: v3(&target_rpy_rates)?,
        };
        let o = self.inner.compute_control(
            dt,
            v3(&cur_pos)?,
            v4(&cur_quat)?,
            v3(&cur_vel)?,
            v3(&cur_ang_vel)?,
            &t,
        );
        Ok((
            PyArray1::from_slice(py, &o.rpm),
            PyArray1::from_slice(py, &o.pos_e),
            o.yaw_e,
        ))
    }

    #[pyo3(signature = (p_coeff_pos=None, i_coeff_pos=None, d_coeff_pos=None, p_coeff_att=None, i_coeff_att=None, d_coeff_att=None))]
    fn set_pid(
        &mut self,
        p_coeff_pos: Option<PyReadonlyArray1<'_, f64>>,
        i_coeff_pos: Option<PyReadonlyArray1<'_, f64>>,
        d_coeff_pos: Option<PyReadonlyArray1<'_, f64>>,
        p_coeff_att: Option<PyReadonlyArray1<'_, f64>>,
        i_coeff_att: Option<PyReadonlyArray1<'_, f64>>,
        d_coeff_att: Option<PyReadonlyArray1<'_, f64>>,
    ) -> PyResult<()> {
        self.inner.set_pid_coefficients(&SetPid {
            p_coeff_pos: opt3(&p_coeff_pos)?,
            i_coeff_pos: opt3(&i_coeff_pos)?,
            d_coeff_pos: opt3(&d_coeff_pos)?,
            p_coeff_att: opt3(&p_coeff_att)?,
            i_coeff_att: opt3(&i_coeff_att)?,
            d_coeff_att: opt3(&d_coeff_att)?,
        });
        Ok(())
    }

    /// Gains as `{P_COEFF_FOR: ..., ...}`.
    fn coefficients<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let c = &self.inner.coeffs;
        let d = PyDict::new(py);
        for (k, v) in [
            ("P_COEFF_FOR", c.p_coeff_for),
            ("I_COEFF_FOR", c.i_coeff_for),
            ("D_COEFF_FOR", c.d_coeff_for),
            ("P_COEFF_TOR", c.p_coeff_tor),
            ("I_COEFF_TOR", c.i_coeff_tor),
            ("D_COEFF_TOR", c.d_coeff_tor),
        ] {
            d.set_item(k, PyArray1::from_slice(py, &v))?;
        }
        d.set_item("GRAVITY", self.inner.gravity)?;
        d.set_item("KF", self.inner.kf)?;
        d.set_item("KM", self.inner.km)?;
        Ok(d)
    }

    #[getter]
    fn control_counter(&self) -> u64 {
        self.inner.control_counter
    }
}

/// `CTBRControl` (quaternions `[w, x, y, z]`).
#[pyclass(module = "dronelab._native")]
pub struct Ctbr {
    inner: CtbrControl,
}

#[pymethods]
impl Ctbr {
    #[new]
    fn new(drone_model: &str, g: f64) -> PyResult<Self> {
        Ok(Ctbr {
            inner: CtbrControl::new(model(drone_model)?, g),
        })
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn compute(
        &mut self,
        cur_pos: PyReadonlyArray1<'_, f64>,
        cur_quat_wxyz: PyReadonlyArray1<'_, f64>,
        cur_vel: PyReadonlyArray1<'_, f64>,
        target_pos: PyReadonlyArray1<'_, f64>,
        target_vel: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<(f64, f64, f64, f64)> {
        let o = self.inner.compute_control(
            v3(&cur_pos)?,
            v4(&cur_quat_wxyz)?,
            v3(&cur_vel)?,
            v3(&target_pos)?,
            v3(&target_vel)?,
        );
        Ok((o[0], o[1], o[2], o[3]))
    }

    #[getter]
    fn constants(&self) -> (f64, f64, f64) {
        (self.inner.gravity, self.inner.kf, self.inner.km)
    }
}

/// `MRACControl`.
#[pyclass(module = "dronelab._native")]
pub struct Mrac {
    inner: MracControl,
}

#[pymethods]
impl Mrac {
    #[new]
    fn new(drone_model: &str, g: f64) -> PyResult<Self> {
        Ok(Mrac {
            inner: MracControl::new(model(drone_model)?, g),
        })
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Returns `(rpm, pos_e, rpy_e)`.
    #[allow(clippy::too_many_arguments)]
    fn compute<'py>(
        &mut self,
        py: Python<'py>,
        dt: f64,
        cur_pos: PyReadonlyArray1<'py, f64>,
        cur_quat: PyReadonlyArray1<'py, f64>,
        cur_vel: PyReadonlyArray1<'py, f64>,
        cur_ang_vel: PyReadonlyArray1<'py, f64>,
        target_pos: PyReadonlyArray1<'py, f64>,
        target_rpy: PyReadonlyArray1<'py, f64>,
        target_vel: PyReadonlyArray1<'py, f64>,
        target_rpy_rates: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<(Arr1<'py>, Arr1<'py>, Arr1<'py>)> {
        let t = Target {
            pos: v3(&target_pos)?,
            rpy: v3(&target_rpy)?,
            vel: v3(&target_vel)?,
            rpy_rates: v3(&target_rpy_rates)?,
        };
        let o = self.inner.compute_control(
            dt,
            v3(&cur_pos)?,
            v4(&cur_quat)?,
            v3(&cur_vel)?,
            v3(&cur_ang_vel)?,
            &t,
        );
        Ok((
            PyArray1::from_slice(py, &o.rpm),
            PyArray1::from_slice(py, &o.pos_e),
            PyArray1::from_slice(py, &o.rpy_e),
        ))
    }

    #[getter]
    fn control_counter(&self) -> u64 {
        self.inner.control_counter
    }
}

/// `BaseControl._getURDFParameter`.
#[pyfunction]
fn urdf_parameter(drone_model: &str, name: &str) -> PyResult<f64> {
    use dronelab::urdf::{UrdfParam, get_urdf_parameter};
    let param = match name {
        "m" => UrdfParam::M,
        "ixx" => UrdfParam::Ixx,
        "iyy" => UrdfParam::Iyy,
        "izz" => UrdfParam::Izz,
        "arm" => UrdfParam::Arm,
        "thrust2weight" => UrdfParam::Thrust2Weight,
        "kf" => UrdfParam::Kf,
        "km" => UrdfParam::Km,
        "max_speed_kmh" => UrdfParam::MaxSpeedKmh,
        "gnd_eff_coeff" => UrdfParam::GndEffCoeff,
        "prop_radius" => UrdfParam::PropRadius,
        "drag_coeff_xy" => UrdfParam::DragCoeffXy,
        "drag_coeff_z" => UrdfParam::DragCoeffZ,
        "dw_coeff_1" => UrdfParam::DwCoeff1,
        "dw_coeff_2" => UrdfParam::DwCoeff2,
        "dw_coeff_3" => UrdfParam::DwCoeff3,
        "length" => UrdfParam::Length,
        "radius" => UrdfParam::Radius,
        "collision_z_offset" => UrdfParam::CollisionZOffset,
        _ => return Err(err(format!("unknown URDF parameter {name:?}"))),
    };
    get_urdf_parameter(model(drone_model)?.urdf(), param).map_err(err)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(urdf_parameter, m)?)?;
    m.add_class::<CompatAviary>()?;
    m.add_class::<CompatBatch>()?;
    m.add_class::<DslPid>()?;
    m.add_class::<Ctbr>()?;
    m.add_class::<Mrac>()?;
    Ok(())
}

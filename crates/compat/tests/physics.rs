mod common;
use common::*;
use dronelab_compat::aviary::Kinematics;
use dronelab_compat::physics::{self, Applied, Body};
use dronelab_compat::{Aviary, AviaryConfig, DroneModel, EnvKind, Params, Physics};
use serde_json::Value;

fn model(v: &Value) -> DroneModel {
    match v.as_str().unwrap() {
        "cf2x" => DroneModel::Cf2x,
        "cf2p" => DroneModel::Cf2p,
        _ => DroneModel::Race,
    }
}

fn aviary_for(case: &Value) -> Aviary {
    let mut cfg = AviaryConfig::new(EnvKind::Ctrl);
    cfg.drone_model = model(&case["model"]);
    cfg.physics = Physics::parse(case["physics"].as_str().unwrap()).unwrap();
    cfg.num_drones = case["num_drones"].as_u64().unwrap() as usize;
    cfg.pyb_freq = case["pyb_freq"].as_u64().unwrap() as u32;
    cfg.ctrl_freq = case["ctrl_freq"].as_u64().unwrap() as u32;
    if !case["initial_xyzs"].is_null() {
        cfg.initial_xyzs = Some(
            case["initial_xyzs"]
                .as_array()
                .unwrap()
                .iter()
                .map(v3)
                .collect(),
        );
    }
    Aviary::new(cfg).unwrap()
}

fn calls_of(v: &Value) -> &Vec<Value> {
    v.as_array().unwrap()
}

fn actions(case: &Value, k: usize) -> Vec<f64> {
    flat(&case["actions"][k])
}

/// The state rows of one recorded observation.
fn rows(obs: &Value) -> Vec<[f64; 20]> {
    obs.as_array()
        .unwrap()
        .iter()
        .map(|r| flat(r).try_into().unwrap())
        .collect()
}

#[test]
fn dyn_trajectories_match_upstream() {
    for case in golden("physics").as_array().unwrap() {
        if case["physics"] != "dyn" {
            continue;
        }
        let mut av = aviary_for(case);
        assert_close(
            "initial obs",
            &av.compute_obs(),
            &flat(&case["obs"][0]),
            1e-15,
        );
        let mut worst = 0.0f64;
        for k in 0..case["actions"].as_array().unwrap().len() {
            let r = av.step(&actions(case, k));
            worst = worst.max(max_rel_err(&r.obs, &flat(&case["obs"][k + 1])));
        }
        println!("dyn {}: worst {worst:e}", case["model"]);
        assert!(worst < 1e-11, "{}: {worst:e}", case["model"]);
    }
}

/// Rebuilds upstream's state at the start of each step from its recorded
/// observation and checks that the first physics substep applies exactly
/// the forces and torques upstream applied.
#[test]
fn pybullet_forces_match_upstream_exactly() {
    let mut checked = 0;
    for case in golden("physics").as_array().unwrap() {
        let phys = Physics::parse(case["physics"].as_str().unwrap()).unwrap();
        if phys == Physics::Dyn {
            continue;
        }
        let p = Params::load(model(&case["model"])).unwrap();
        let n = case["num_drones"].as_u64().unwrap() as usize;
        let substeps =
            (case["pyb_freq"].as_u64().unwrap() / case["ctrl_freq"].as_u64().unwrap()) as usize;
        for (k, recorded) in case["forces"].as_array().unwrap().iter().enumerate() {
            let state = rows(&case["obs"][k]);
            let kin: Vec<Kinematics> = state
                .iter()
                .map(|s| Kinematics {
                    pos: [s[0], s[1], s[2]],
                    quat: [s[3], s[4], s[5], s[6]],
                    rpy: [s[7], s[8], s[9]],
                    vel: [s[10], s[11], s[12]],
                    ang_v: [s[13], s[14], s[15]],
                })
                .collect();
            let act = actions(case, k);
            let positions: Vec<_> = kin.iter().map(|k| k.pos).collect();
            let mut ours = Vec::new();
            for i in 0..n {
                let rpm: [f64; 4] = act[i * 4..i * 4 + 4].try_into().unwrap();
                let last: [f64; 4] = state[i][16..20].try_into().unwrap();
                let body = Body::new(kin[i].pos, kin[i].quat);
                let mut f = Vec::new();
                physics::substep_forces(
                    &p,
                    phys,
                    &rpm,
                    &last,
                    &body,
                    kin[i].quat,
                    kin[i].rpy,
                    kin[i].vel,
                    &positions,
                    i,
                    &mut f,
                );
                ours.extend(f.into_iter().filter(|a| *a != Applied::ForwardKinematics));
            }
            // Upstream's calls for the first substep only.
            // A substep starts with drone 0's first rotor force; ground
            // effect also starts with a link-0 force, but right after the
            // same drone's yaw torque.
            let calls = calls_of(recorded);
            let body0 = calls[0][1].as_u64().unwrap();
            let has_ground = matches!(phys, Physics::PybGnd | Physics::PybGndDragDw);
            let starts_substep = |j: usize| {
                let c = &calls[j];
                let is_torque = |m: usize| {
                    m < calls.len()
                        && calls[m][0] == "torque"
                        && calls[m][1].as_u64() == Some(body0)
                };
                // Ground-effect forces follow the yaw torque; a new substep's
                // rotor forces are themselves followed by a yaw torque.
                let ground_effect = has_ground && j > 0 && is_torque(j - 1) && !is_torque(j + 4);
                c[0] == "force" && c[1].as_u64() == Some(body0) && c[2] == 0 && !ground_effect
            };
            let end = (1..calls.len())
                .find(|&j| starts_substep(j))
                .unwrap_or(calls.len());
            assert!(substeps == 1 || end < calls.len());
            let first = &calls[..end];
            assert_eq!(
                first.len(),
                ours.len(),
                "{} step {k}: number of applied forces",
                case["physics"]
            );
            for (call, a) in first.iter().zip(&ours) {
                let (kind, link, v) = match a {
                    Applied::Force { link, f } => ("force", *link, f),
                    Applied::Torque { link, t } => ("torque", *link, t),
                    Applied::ForwardKinematics => unreachable!(),
                };
                assert_eq!(
                    call[0].as_str().unwrap(),
                    kind,
                    "{} step {k}",
                    case["physics"]
                );
                assert_eq!(
                    call[2].as_u64().unwrap() as usize,
                    link,
                    "{} {} step {k}: {first:?}",
                    case["physics"],
                    case["num_drones"]
                );
                assert_close(
                    &format!("{} step {k} {kind}", case["physics"]),
                    v,
                    &flat(&call[3]),
                    1e-13,
                );
                checked += 1;
            }
        }
    }
    println!("checked {checked} forces/torques");
    assert!(checked > 1000);
}

#[test]
fn pybullet_rigid_body_step_tracks_upstream() {
    println!(
        "{:22} {:>6} {:>12} {:>12} {:>12}",
        "case", "steps", "pos err m", "vel err m/s", "rpy err rad"
    );
    for case in golden("physics").as_array().unwrap() {
        if case["physics"] == "dyn" {
            continue;
        }
        let mut av = aviary_for(case);
        let n_steps = case["actions"].as_array().unwrap().len();
        let (mut pe, mut ve, mut ae) = (0.0f64, 0.0f64, 0.0f64);
        let mut compared = 0;
        for k in 0..n_steps {
            av.step(&actions(case, k));
            if av.contacts > 0 {
                break; // PyBullet's contact solver is not reproduced
            }
            let up = rows(&case["obs"][k + 1]);
            for (i, s) in up.iter().enumerate() {
                let o = av.state_vector(i);
                let d =
                    |a: usize| ((0..3).map(|j| (o[a + j] - s[a + j]).powi(2)).sum::<f64>()).sqrt();
                pe = pe.max(d(0));
                ve = ve.max(d(10));
                ae = ae.max(d(7));
            }
            compared = k + 1;
        }
        let label = format!(
            "{} {} x{}",
            case["physics"].as_str().unwrap(),
            case["model"].as_str().unwrap(),
            case["num_drones"]
        );
        println!("{label:22} {compared:>6} {pe:>12.3e} {ve:>12.3e} {ae:>12.3e}");
        assert!(compared > 0);
        // One second of free flight: within 1 mm, 1 cm/s, 1 mrad.
        assert!(pe < 1e-3 && ve < 1e-2 && ae < 1e-3, "{label}");
    }
}

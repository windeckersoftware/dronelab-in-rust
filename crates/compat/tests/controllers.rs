mod common;
use common::*;
use dronelab_compat::control::*;
use dronelab_compat::{DroneModel, G};
use serde_json::Value;

fn model(v: &Value) -> DroneModel {
    match v.as_str().unwrap() {
        "cf2x" => DroneModel::Cf2x,
        "cf2p" => DroneModel::Cf2p,
        _ => DroneModel::Race,
    }
}

#[test]
fn dsl_pid_matches_upstream_step_by_step() {
    let g = golden("controllers");
    for case in g["dsl"].as_array().unwrap() {
        let mut c = DslPidControl::new(model(&case["model"]), G).unwrap();
        let dt = f(&case["dt"]);
        let mut worst = 0.0f64;
        for (k, step) in case["steps"].as_array().unwrap().iter().enumerate() {
            let t = Target {
                pos: v3(&case["target_pos"][k]),
                rpy: v3(&case["target_rpy"][k]),
                vel: v3(&case["target_vel"][k]),
                rpy_rates: v3(&case["target_rpy_rates"][k]),
            };
            let o = c.compute_control(
                dt,
                v3(&case["pos"][k]),
                v4(&case["quat"][k]),
                v3(&case["vel"][k]),
                v3(&case["ang_vel"][k]),
                &t,
            );
            let ours: Vec<f64> = o
                .rpm
                .iter()
                .chain(&o.pos_e)
                .copied()
                .chain([o.yaw_e])
                .collect();
            let up: Vec<f64> = flat(&step["rpm"])
                .into_iter()
                .chain(flat(&step["pos_e"]))
                .chain([f(&step["yaw_e"])])
                .collect();
            worst = worst.max(max_rel_err(&ours, &up));
        }
        println!("dsl {} {}: worst {worst:e}", case["model"], case["case"]);
        assert!(
            worst < 1e-9,
            "{} {}: {worst:e}",
            case["model"],
            case["case"]
        );

        c.set_pid_coefficients(&SetPid {
            p_coeff_pos: Some([0.5, 0.5, 1.0]),
            ..Default::default()
        });
        c.reset();
        let o = c.compute_control(
            dt,
            v3(&case["pos"][0]),
            v4(&case["quat"][0]),
            v3(&case["vel"][0]),
            v3(&case["ang_vel"][0]),
            &Target {
                pos: v3(&case["target_pos"][0]),
                ..Default::default()
            },
        );
        assert_close(
            "after setPIDCoefficients + reset",
            &o.rpm,
            &flat(&case["after_set_pid_and_reset"]["rpm"]),
            1e-12,
        );
    }
    assert!(DslPidControl::new(DroneModel::Race, G).is_err());
}

#[test]
fn mrac_matches_upstream_step_by_step() {
    let g = golden("controllers");
    for case in g["mrac"].as_array().unwrap() {
        let mut c = MracControl::new(model(&case["model"]), G);
        assert_close("Kx", &c.kx.concat(), &flat(&case["Kx"]), 0.0);
        let dt = f(&case["dt"]);
        let mut worst = 0.0f64;
        for (k, step) in case["steps"].as_array().unwrap().iter().enumerate() {
            let t = Target {
                pos: v3(&case["target_pos"][k]),
                ..Default::default()
            };
            let o = c.compute_control(
                dt,
                v3(&case["pos"][k]),
                v4(&case["quat"][k]),
                v3(&case["vel"][k]),
                v3(&case["ang_vel"][k]),
                &t,
            );
            let ours: Vec<f64> = o
                .rpm
                .iter()
                .chain(&o.pos_e)
                .chain(&o.rpy_e)
                .copied()
                .collect();
            let up: Vec<f64> = [
                flat(&step["rpm"]),
                flat(&step["pos_e"]),
                flat(&step["rpy_e"]),
            ]
            .concat();
            worst = worst.max(max_rel_err(&ours, &up));
        }
        println!("mrac {}: worst {worst:e}", case["model"]);
        assert!(worst < 1e-9, "{}: {worst:e}", case["model"]);
    }
}

#[test]
fn ctbr_matches_upstream() {
    let g = golden("controllers");
    for case in g["ctbr"].as_array().unwrap() {
        let mut c = CtbrControl::new(model(&case["model"]), G);
        let mut worst = 0.0f64;
        for (k, up) in case["out"].as_array().unwrap().iter().enumerate() {
            let o = c.compute_control(
                v3(&case["pos"][k]),
                v4(&case["quat_wxyz"][k]),
                v3(&case["vel"][k]),
                v3(&case["target_pos"][k]),
                v3(&case["target_vel"][k]),
            );
            worst = worst.max(max_rel_err(&o, &flat(up)));
        }
        println!("ctbr {}: worst {worst:e}", case["model"]);
        assert!(worst < 1e-12, "{}: {worst:e}", case["model"]);
    }
}

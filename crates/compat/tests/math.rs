mod common;
use common::*;
use dronelab_compat::math::*;
use dronelab_compat::{DroneModel, G, Params};

#[test]
fn urdf_constants_match_upstream() {
    let g = golden("params");
    for model in [DroneModel::Cf2x, DroneModel::Cf2p, DroneModel::Race] {
        let up = &g[model.value()];
        let p = Params::load(model).unwrap();
        let pairs: [(&str, Vec<f64>); 24] = [
            ("M", vec![p.m]),
            ("L", vec![p.l]),
            ("THRUST2WEIGHT_RATIO", vec![p.thrust2weight]),
            ("J", p.j.iter().flatten().copied().collect()),
            ("J_INV", p.j_inv.iter().flatten().copied().collect()),
            ("KF", vec![p.kf]),
            ("KM", vec![p.km]),
            ("COLLISION_H", vec![p.collision_h]),
            ("COLLISION_R", vec![p.collision_r]),
            ("COLLISION_Z_OFFSET", vec![p.collision_z_offset]),
            ("MAX_SPEED_KMH", vec![p.max_speed_kmh]),
            ("GND_EFF_COEFF", vec![p.gnd_eff_coeff]),
            ("PROP_RADIUS", vec![p.prop_radius]),
            ("DRAG_COEFF", p.drag_coeff.to_vec()),
            ("DW_COEFF_1", vec![p.dw_coeff_1]),
            ("DW_COEFF_2", vec![p.dw_coeff_2]),
            ("DW_COEFF_3", vec![p.dw_coeff_3]),
            ("GRAVITY", vec![p.gravity]),
            ("HOVER_RPM", vec![p.hover_rpm]),
            ("MAX_RPM", vec![p.max_rpm]),
            ("MAX_THRUST", vec![p.max_thrust]),
            ("MAX_XY_TORQUE", vec![p.max_xy_torque]),
            ("MAX_Z_TORQUE", vec![p.max_z_torque]),
            ("GND_EFF_H_CLIP", vec![p.gnd_eff_h_clip]),
        ];
        for (k, ours) in pairs {
            assert_eq!(ours, flat(&up[k]), "{} {k}", model.value());
        }
        assert_eq!(
            p.default_init_xyzs(3).concat(),
            flat(&up["INIT_XYZS"]),
            "{} INIT_XYZS",
            model.value()
        );
        assert_eq!(G, f(&up["G"]));
    }
}

#[test]
fn rotation_helpers_match_pybullet_scipy_transforms3d() {
    let g = golden("math");
    let quats = g["quat_xyzw"].as_array().unwrap();
    let mut worst = [0.0f64; 7];
    let mut track =
        |slot: usize, a: &[f64], b: &[f64]| worst[slot] = worst[slot].max(max_rel_err(a, b));
    for (n, q) in quats.iter().enumerate() {
        let q = v4(q);
        track(0, &pb_matrix(q).concat(), &flat(&g["pb_matrix"][n]));
        track(1, &pb_euler(q), &flat(&g["pb_euler"][n]));
        let m = mat3(&g["scipy_quat_to_matrix"][n]);
        track(
            2,
            &scipy_quat_to_matrix(q).concat(),
            &flat(&g["scipy_quat_to_matrix"][n]),
        );
        track(
            3,
            &scipy_matrix_to_euler_xyz(&m),
            &flat(&g["scipy_matrix_to_euler_XYZ"][n]),
        );
        track(4, &t3d_mat2quat(&m), &flat(&g["t3d_mat2quat_wxyz"][n]));
        let wxyz = [q[3], q[0], q[1], q[2]];
        track(
            5,
            &t3d_rotate_vector(v3(&g["vec"][n]), wxyz),
            &flat(&g["t3d_rotate_vector_wxyz"][n]),
        );
        let a = wxyz;
        let b = v4(&quats[quats.len() - 1 - n]);
        track(
            6,
            &t3d_qmult(t3d_qconjugate(a), [b[3], b[0], b[1], b[2]]),
            &flat(&g["t3d_qmult_conj"][n]),
        );
    }
    for (n, e) in g["euler"].as_array().unwrap().iter().enumerate() {
        let e = v3(e);
        assert_close(
            "pb_quat_from_euler",
            &pb_quat_from_euler(e),
            &flat(&g["pb_quat_from_euler"][n]),
            1e-15,
        );
        assert_close(
            "scipy euler->quat",
            &scipy_euler_xyz_to_quat(e),
            &flat(&g["scipy_euler_XYZ_to_quat"][n]),
            1e-15,
        );
        assert_close(
            "scipy euler->matrix",
            &scipy_euler_xyz_to_matrix(e).concat(),
            &flat(&g["scipy_euler_XYZ_to_matrix"][n]),
            1e-15,
        );
    }
    let names = [
        "pb_matrix",
        "pb_euler",
        "scipy as_matrix",
        "scipy as_euler XYZ",
        "t3d mat2quat",
        "t3d rotate_vector",
        "t3d qmult",
    ];
    for (name, w) in names.iter().zip(worst) {
        println!("{name:20} worst {w:e}");
        assert!(w < 1e-14, "{name}: {w:e}");
    }
}

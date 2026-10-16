//! Whole episodes of upstream's RL environments, step by step.
mod common;
use common::*;
use dronelab_compat::{ActionType, Aviary, AviaryConfig, EnvKind, Physics};
use serde_json::Value;

fn make(case: &Value) -> Aviary {
    let act = ActionType::parse(case["act"].as_str().unwrap()).unwrap();
    let kind = match case["env"].as_str().unwrap() {
        "HoverAviary" => EnvKind::Hover(act),
        "MultiHoverAviary" => EnvKind::MultiHover(act),
        "VelocityAviary" => EnvKind::Velocity,
        other => panic!("{other}"),
    };
    let mut cfg = AviaryConfig::new(kind);
    cfg.physics = Physics::parse(case["physics"].as_str().unwrap()).unwrap();
    if let Some(n) = case["kwargs"]["num_drones"].as_u64() {
        cfg.num_drones = n as usize;
    }
    if let Some(f) = case["kwargs"]["ctrl_freq"].as_u64() {
        cfg.ctrl_freq = f as u32;
    }
    Aviary::new(cfg).unwrap()
}

#[test]
fn rl_episodes_match_upstream() {
    println!(
        "{:18} {:5} {:10} {:>5} {:>11} {:>11}",
        "env", "phys", "act", "steps", "obs err", "reward err"
    );
    println!(
        "(tolerance follows upstream's own drift when KF changes by 1e-12; 'chaotic' steps are beyond any comparison)"
    );
    for case in golden("envs").as_array().unwrap() {
        let mut av = make(case);
        let label = format!(
            "{:18} {:5} {:10}",
            case["env"].as_str().unwrap(),
            case["physics"].as_str().unwrap(),
            case["act"].as_str().unwrap()
        );
        assert_close(
            &format!("{label} initial obs"),
            &av.reset(),
            &flat(&case["initial_obs"]),
            1e-7,
        );
        let (mut oe, mut re, mut n) = (0.0f64, 0.0f64, 0usize);
        // DYN is ported operation for operation, so differences start at
        // rounding level. PYB stands in for Bullet's integrator; open loop
        // it drifts from Bullet up to ~1e4 times more than a 1e-12 change of
        // KF does (most in attitude, which KF barely touches).
        let (floor, amplification) = if case["physics"] == "dyn" {
            (1e-9, 1e2)
        } else {
            (1e-5, 1e4)
        };
        // Upstream with KF changed by one part in 1e12 drifts by this much:
        // the episode's own sensitivity to tiny per-step errors.
        let envelope: Vec<f64> = vec(&case["self_divergence"])
            .iter()
            .scan(0.0f64, |m, &d| {
                *m = m.max(d);
                Some(*m)
            })
            .collect();
        let mut chaotic = 0;
        for (k, step) in case["steps"].as_array().unwrap().iter().enumerate() {
            let allowed = floor + amplification * envelope[k];
            if allowed > 1e-3 {
                chaotic = case["steps"].as_array().unwrap().len() - k;
                break; // deep in the chaotic part; nothing can be compared
            }
            let r = av.step(&flat(&step["action"]));
            if av.contacts > 0 {
                break; // ground contact is only approximate
            }
            let up = flat(&step["obs"]);
            // Upstream rounds the kinematic part of RL observations to
            // float32, so values 1e-16 apart can land a float32 step apart.
            let err = r
                .obs
                .iter()
                .zip(&up)
                .map(|(a, b)| ((a - b).abs() - b.abs() * f32::EPSILON as f64).max(0.0))
                .fold(0.0, f64::max);
            assert!(
                err <= allowed,
                "{label} step {k}: error {err:e} > allowed {allowed:e}"
            );
            oe = oe.max(err);
            re = re.max((r.reward - f(&step["reward"])).abs());
            if envelope[k] <= floor {
                assert_eq!(
                    r.terminated,
                    step["terminated"].as_bool().unwrap(),
                    "{label} terminated at step {k}"
                );
                assert_eq!(
                    r.truncated,
                    step["truncated"].as_bool().unwrap(),
                    "{label} truncated at step {k}"
                );
            }
            n += 1;
        }
        println!(
            "{label} {n:>5} {oe:>11.2e} {re:>11.2e}   chaotic steps {chaotic}, upstream self-drift {:.1e}",
            envelope[n.saturating_sub(1)]
        );
        assert!(n >= 5, "{label}: only {n} steps before contact");
    }
}

#[test]
fn episode_length_matches_upstream() {
    let case = golden("envs")
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["physics"] == "dyn" && c["act"] == "one_d_rpm" && c["env"] == "HoverAviary")
        .unwrap()
        .clone();
    let steps = case["steps"].as_array().unwrap();
    let mut av = make(&case);
    av.reset();
    let mut len = 0;
    for step in steps {
        len += 1;
        if av.step(&flat(&step["action"])).truncated {
            break;
        }
    }
    assert_eq!(len, steps.len(), "episode length");
    println!("upstream episode length {len}");
}

/// Upstream (checked against v2.2.0): a time-limited HoverAviary episode at
/// 240/30 Hz lasts 242 steps, and `reset()` keeps the PID integrals and the
/// action buffer.
#[test]
fn upstream_episode_and_reset_semantics() {
    let mut cfg = AviaryConfig::new(EnvKind::Hover(ActionType::OneDRpm));
    cfg.physics = Physics::Dyn;
    let mut av = Aviary::new(cfg).unwrap();
    av.reset();
    let mut n = 0;
    loop {
        n += 1;
        let r = av.step(&[0.0]);
        if r.terminated || r.truncated {
            assert!(r.truncated);
            break;
        }
    }
    assert_eq!(n, 242);

    let mut cfg = AviaryConfig::new(EnvKind::Hover(ActionType::Pid));
    cfg.physics = Physics::Dyn;
    let mut av = Aviary::new(cfg).unwrap();
    av.reset();
    for _ in 0..5 {
        av.step(&[0.3, 0.2, 0.8]);
    }
    let before = av.ctrl[0].clone();
    let obs = av.reset();
    assert_eq!(
        format!("{:?}", av.ctrl[0]),
        format!("{before:?}"),
        "controller state kept"
    );
    assert!(obs[12..].iter().any(|&x| x != 0.0), "action buffer kept");
}

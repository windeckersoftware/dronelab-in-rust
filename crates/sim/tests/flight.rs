//! The shared PID controller flies every airframe with full realism.

use std::sync::Arc;

use dronelab::{ActionMode, Airframe, Env, EnvConfig, Randomization, Realism};

/// Flies the expert for `seconds` and returns (crashed, final distance / pos_scale).
fn fly(cfg: &Arc<EnvConfig>, seed: u64) -> (bool, f32) {
    let mut env = Env::new(Arc::clone(cfg), seed);
    let mut obs = vec![0.0; cfg.obs_dim()];
    let mut act = vec![0.0; cfg.act_dim()];
    for _ in 0..cfg.episode_steps() {
        env.expert_action(&mut act).unwrap();
        if env.step(&act, &mut obs).terminated {
            return (true, f32::INFINITY);
        }
    }
    (
        false,
        (env.target - env.drone.state.pos).norm() / cfg.pos_scale,
    )
}

#[test]
fn pid_reaches_targets_with_full_realism() {
    for name in Airframe::PRESETS {
        for action in [ActionMode::RotorThrust, ActionMode::Velocity] {
            let mut cfg = EnvConfig::reach(Airframe::preset(name).unwrap());
            cfg.realism = Realism::FULL;
            cfg.randomization = Randomization::NONE;
            cfg.action = action;
            cfg.episode_seconds = 25.0;
            let cfg = Arc::new(cfg);
            for seed in 0..8 {
                let (crashed, d) = fly(&cfg, seed);
                assert!(!crashed, "{name} {action:?} seed {seed} crashed");
                assert!(d < 0.2, "{name} {action:?} seed {seed}: {d}");
            }
        }
    }
}

#[test]
fn pid_survives_randomized_single_seat_evtol() {
    let af = Airframe::evtol_single_seat();
    let mut cfg = EnvConfig::reach(af.clone());
    cfg.randomization = Randomization::default_for(&af);
    cfg.episode_seconds = 25.0;
    let cfg = Arc::new(cfg);
    for seed in 0..16 {
        let (crashed, d) = fly(&cfg, seed);
        assert!(!crashed && d < 0.2, "seed {seed}: crashed {crashed}, {d}");
    }
}

#[test]
fn racer_completes_timed_laps() {
    let mut cfg = EnvConfig::race(Airframe::racer());
    cfg.action = ActionMode::Velocity;
    cfg.episode_seconds = 40.0;
    let cfg = Arc::new(cfg);
    let mut env = Env::new(Arc::clone(&cfg), 1);
    let mut obs = vec![0.0; cfg.obs_dim()];
    let mut act = vec![0.0; cfg.act_dim()];
    let mut laps = Vec::new();
    for _ in 0..cfg.episode_steps() {
        env.expert_action(&mut act).unwrap();
        let info = env.step(&act, &mut obs);
        assert!(!info.terminated, "crashed at {:.1} s", env.time());
        laps.extend(info.lap_time);
    }
    assert!(laps.len() >= 3, "{laps:?}");
    assert_eq!(laps, env.lap.laps);
    assert!(laps.iter().all(|&t| t > 1.0 && t < 15.0), "{laps:?}");
}

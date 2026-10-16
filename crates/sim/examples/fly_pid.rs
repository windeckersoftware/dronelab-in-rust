//! Flies the built-in PID controller on every airframe and prints how close
//! it gets to the target.
//!
//! cargo run --release -p dronelab --example fly_pid

use dronelab::{ActionMode, Airframe, Env, EnvConfig, Randomization, Realism};
use std::sync::Arc;

fn main() {
    for name in Airframe::PRESETS {
        for (label, realism, random) in [
            ("ideal", Realism::IDEAL, false),
            ("full", Realism::FULL, true),
        ] {
            for action in [ActionMode::RotorThrust, ActionMode::Velocity] {
                let af = Airframe::preset(name).unwrap();
                let mut cfg = EnvConfig::reach(af);
                cfg.realism = realism;
                cfg.action = action;
                if !random {
                    cfg.randomization = Randomization::NONE;
                }
                let cfg = Arc::new(cfg);
                let (mut worst, mut crashes, mut total) = (0.0f32, 0, 0.0f32);
                let seeds = 20;
                for seed in 0..seeds {
                    let mut env = Env::new(Arc::clone(&cfg), seed);
                    let mut obs = vec![0.0; cfg.obs_dim()];
                    let mut act = vec![0.0; cfg.act_dim()];
                    let mut crashed = false;
                    for _ in 0..cfg.episode_steps() {
                        env.expert_action(&mut act).unwrap();
                        let info = env.step(&act, &mut obs);
                        if info.terminated {
                            crashed = true;
                            break;
                        }
                    }
                    let d = (env.target - env.drone.state.pos).norm() / cfg.pos_scale;
                    if crashed {
                        crashes += 1
                    } else {
                        worst = worst.max(d);
                        total += d
                    }
                }
                println!(
                    "{name:18} {label:5} {action:?}: crashes {crashes}/{seeds}, final dist/pos_scale mean {:.3} worst {:.3}",
                    total / (seeds - crashes).max(1) as f32,
                    worst
                );
            }
        }
    }
}

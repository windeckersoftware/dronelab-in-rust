//! Flies the built-in PID controller around each airframe's default circuit.
//!
//! cargo run --release -p dronelab --example race_pid

use dronelab::{ActionMode, Airframe, Env, EnvConfig};
use std::sync::Arc;

fn main() {
    for name in Airframe::PRESETS {
        let mut cfg = EnvConfig::race(Airframe::preset(name).unwrap());
        cfg.action = ActionMode::Velocity;
        cfg.episode_seconds = 120.0;
        let cfg = Arc::new(cfg);
        let mut env = Env::new(Arc::clone(&cfg), 1);
        let mut obs = vec![0.0; cfg.obs_dim()];
        let mut act = vec![0.0; cfg.act_dim()];
        let mut crashed = false;
        for _ in 0..cfg.episode_steps() {
            env.expert_action(&mut act).unwrap();
            let info = env.step(&act, &mut obs);
            if info.done() {
                crashed = info.crashed;
                break;
            }
        }
        let best = env
            .lap
            .best_lap()
            .map_or("-".into(), |t| format!("{t:.2} s"));
        println!(
            "{name:18} gates {:3}  laps {}  best lap {best}  crashed {crashed}",
            env.lap.gates_passed,
            env.lap.laps.len()
        );
    }
}

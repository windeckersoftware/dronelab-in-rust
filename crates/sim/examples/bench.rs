//! Steps per second, single thread and all cores.
//!
//! cargo run --release -p dronelab --example bench
//!
//! "physics steps/s" counts vehicle updates (env steps x substeps), the
//! number to compare with PyBullet's `pyb_freq` stepping.

use std::time::Instant;

use dronelab::{Airframe, BatchEnv, EnvConfig, Realism};

fn rate(cfg: &EnvConfig, envs: usize, threads: Option<usize>) -> f64 {
    let mut batch = BatchEnv::new(cfg.clone(), envs, 0, threads).unwrap();
    let actions = vec![0.0f32; envs * cfg.act_dim()];
    let steps = (200_000 / envs).clamp(20, 2000);
    batch.step(&actions); // warm up
    let t = Instant::now();
    for _ in 0..steps {
        batch.step(&actions);
    }
    (steps * envs) as f64 / t.elapsed().as_secs_f64()
}

fn main() {
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    println!("{cores} cores; physics {} Hz, control {} Hz", 240, 30);
    println!(
        "{:18} {:6} {:>6} {:>14} {:>16}",
        "airframe", "physics", "envs", "env steps/s", "physics steps/s"
    );
    for name in ["crazyflie_x", "evtol_two_seat"] {
        for (label, realism) in [("ideal", Realism::IDEAL), ("full", Realism::FULL)] {
            let mut cfg = EnvConfig::reach(Airframe::preset(name).unwrap());
            cfg.realism = realism;
            let sub = cfg.substeps() as f64;
            for (envs, threads) in [(1, Some(1)), (1024, Some(1)), (1024, None), (8192, None)] {
                let r = rate(&cfg, envs, threads);
                let t = threads.map_or(format!("{cores}t"), |n| format!("{n}t"));
                println!(
                    "{name:18} {label:6} {:>6} {:>14.0} {:>16.0}",
                    format!("{envs}/{t}"),
                    r,
                    r * sub
                );
            }
        }
    }
}

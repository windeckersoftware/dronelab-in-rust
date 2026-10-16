//! Flies the single-seat eVTOL to a target with the built-in PID controller
//! and writes the flight as an ArduPilot DataFlash log and a PX4 ULog.
//!
//! cargo run --release -p dronelab --example export_log [out_dir]

use std::path::PathBuf;
use std::sync::Arc;

use dronelab::{Airframe, Env, EnvConfig, GeoOrigin, LogFormat};

fn main() -> std::io::Result<()> {
    let dir = PathBuf::from(std::env::args().nth(1).unwrap_or_else(|| ".".into()));
    let cfg = Arc::new(EnvConfig::reach(Airframe::evtol_single_seat()));
    let mut env = Env::new(Arc::clone(&cfg), 7);
    env.start_log(GeoOrigin::default());
    let mut obs = vec![0.0; cfg.obs_dim()];
    let mut act = vec![0.0; cfg.act_dim()];
    for _ in 0..cfg.episode_steps() {
        env.expert_action(&mut act).unwrap();
        if env.step(&act, &mut obs).done() {
            break;
        }
    }
    let log = env.stop_log().unwrap();
    for (name, format) in [
        ("flight.bin", LogFormat::DataFlash),
        ("flight.ulg", LogFormat::ULog),
    ] {
        let path = dir.join(name);
        log.save(&path, format)?;
        println!(
            "{}: {} samples, {:.1} s",
            path.display(),
            log.len(),
            log.samples.last().unwrap().time
        );
    }
    Ok(())
}

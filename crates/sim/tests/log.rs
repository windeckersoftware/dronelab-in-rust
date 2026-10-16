//! Flight log export: structure checks that do not need the Python readers
//! (crates/py/tests/test_logs.py parses the files with pyulog and pymavlink).

use std::sync::Arc;

use dronelab::{Airframe, Env, EnvConfig, GeoOrigin};

fn fly(af: Airframe, steps: usize) -> dronelab::FlightLog {
    let mut cfg = EnvConfig::reach(af);
    cfg.episode_seconds = 1.0;
    let cfg = Arc::new(cfg);
    let mut env = Env::new(Arc::clone(&cfg), 3);
    env.start_log(GeoOrigin::default());
    let mut obs = vec![0.0; cfg.obs_dim()];
    let mut act = vec![0.0; cfg.act_dim()];
    for _ in 0..steps {
        env.expert_action(&mut act).unwrap();
        if env.step(&act, &mut obs).done() {
            env.reset(None);
        }
    }
    env.stop_log().unwrap()
}

#[test]
fn time_continues_across_episodes() {
    let log = fly(Airframe::evtol_single_seat(), 75);
    // Initial state, 75 steps, plus the first state of the two new episodes.
    assert_eq!(log.len(), 78);
    assert_eq!(log.episodes.len(), 3);
    let dt = 1.0 / 30.0;
    for w in log.samples.windows(2) {
        let step = w[1].time - w[0].time;
        assert!((step - dt).abs() < 1e-9, "{step}");
    }
    assert!(log.samples.last().unwrap().used_mah > 0.0);
}

#[test]
fn dataflash_messages_are_framed() {
    let log = fly(Airframe::evtol_two_seat(), 10);
    let mut buf = Vec::new();
    log.write_dataflash(&mut buf).unwrap();
    // Walk the file using the lengths from its own FMT records.
    let mut len = [0usize; 256];
    len[128] = 89;
    let mut counts = [0usize; 256];
    let mut i = 0;
    while i < buf.len() {
        assert_eq!(&buf[i..i + 2], &[0xA3, 0x95], "header at {i}");
        let id = buf[i + 2] as usize;
        if id == 128 {
            len[buf[i + 3] as usize] = buf[i + 4] as usize;
        }
        assert!(len[id] > 0, "unknown message {id} at {i}");
        counts[id] += 1;
        i += len[id];
    }
    assert_eq!(i, buf.len());
    // 16 rotors: RCOU and RCO2 but no RCO3; one ESC message per rotor.
    let (att, rcou, rco2, rco3, esc) = (131, 136, 137, 138, 139);
    assert_eq!(counts[att], 11);
    assert_eq!(counts[rcou], 11);
    assert_eq!(counts[rco2], 11);
    assert_eq!(counts[rco3], 0);
    assert_eq!(counts[esc], 16 * 11);
}

#[test]
fn ulog_messages_are_framed() {
    let log = fly(Airframe::racer(), 10);
    let mut buf = Vec::new();
    log.write_ulog(&mut buf).unwrap();
    assert_eq!(&buf[..7], b"ULog\x01\x12\x35");
    let mut i = 16;
    let mut data = 0;
    let mut kinds = Vec::new();
    while i < buf.len() {
        let size = u16::from_le_bytes([buf[i], buf[i + 1]]) as usize;
        let kind = buf[i + 2];
        kinds.push(kind);
        if kind == b'D' {
            data += 1;
        }
        i += 3 + size;
    }
    assert_eq!(i, buf.len());
    assert_eq!(kinds[0], b'B', "flag bits come first");
    // 7 topics (one actuator_outputs instance for 4 rotors) x 11 samples.
    assert_eq!(data, 7 * 11);
}

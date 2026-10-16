//! Batch stepping, determinism and auto-reset.

use dronelab::rng::Rng;
use dronelab::{ActionMode, Airframe, BatchEnv, EnvConfig};

fn random_actions(rng: &mut Rng, n: usize) -> Vec<f32> {
    (0..n).map(|_| rng.range(-0.3, 0.3)).collect()
}

/// Runs `steps` steps with seeded random actions and returns every observation.
fn rollout(
    cfg: &EnvConfig,
    envs: usize,
    seed: u64,
    threads: Option<usize>,
    steps: usize,
) -> Vec<f32> {
    let mut batch = BatchEnv::new(cfg.clone(), envs, seed, threads).unwrap();
    let mut rng = Rng::new(99);
    let mut all = batch.obs.clone();
    for _ in 0..steps {
        let a = random_actions(&mut rng, envs * cfg.act_dim());
        batch.step(&a);
        all.extend_from_slice(&batch.obs);
        all.extend(batch.infos.iter().map(|i| i.reward));
    }
    all
}

#[test]
fn same_seed_same_flight_regardless_of_threads() {
    for name in ["crazyflie_x", "evtol_two_seat"] {
        let cfg = EnvConfig::reach(Airframe::preset(name).unwrap());
        let one = rollout(&cfg, 64, 7, Some(1), 120);
        let many = rollout(&cfg, 64, 7, Some(8), 120);
        let again = rollout(&cfg, 64, 7, None, 120);
        // Bit-for-bit, including through crashes and automatic resets.
        assert!(
            one.iter()
                .zip(&many)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "{name}: threads changed the result"
        );
        assert!(
            one.iter()
                .zip(&again)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "{name}: rerun changed the result"
        );
        assert_ne!(
            one,
            rollout(&cfg, 64, 8, None, 120),
            "{name}: different seeds should differ"
        );
    }
}

#[test]
fn env_result_does_not_depend_on_batch_size() {
    let cfg = EnvConfig::race(Airframe::racer());
    let mut small = BatchEnv::new(cfg.clone(), 1, 3, None).unwrap();
    let mut big = BatchEnv::new(cfg.clone(), 300, 3, None).unwrap();
    let ad = cfg.act_dim();
    let mut rng = Rng::new(5);
    for _ in 0..100 {
        let a = random_actions(&mut rng, ad);
        small.step(&a);
        let mut wide = vec![0.0; 300 * ad];
        wide[..ad].copy_from_slice(&a);
        big.step(&wide);
        assert_eq!(small.obs[..], big.obs[..cfg.obs_dim()]);
    }
}

#[test]
fn finished_envs_reset_and_keep_their_last_observation() {
    let mut cfg = EnvConfig::reach(Airframe::evtol_single_seat());
    cfg.episode_seconds = 1.0;
    let mut batch = BatchEnv::new(cfg.clone(), 4, 0, None).unwrap();
    let od = cfg.obs_dim();
    let zeros = vec![0.0; 4 * cfg.act_dim()];
    for step in 1..=cfg.episode_steps() {
        let before = batch.obs.clone();
        batch.step(&zeros);
        let done = batch.infos[0].done();
        assert_eq!(done, step == cfg.episode_steps(), "step {step}");
        if done {
            assert!(batch.infos.iter().all(|i| i.truncated && !i.terminated));
            // The terminal observation continues the old episode; the new
            // observation starts a fresh one somewhere else.
            let (t, o) = (&batch.terminal_obs[..od], &batch.obs[..od]);
            assert!((t[0] - before[0]).abs() < 0.2);
            assert_ne!(t, o);
        }
    }
}

#[test]
fn action_modes_have_expected_sizes() {
    let mut cfg = EnvConfig::reach(Airframe::evtol_two_seat());
    assert_eq!((cfg.act_dim(), cfg.obs_dim()), (16, 34));
    cfg.action = ActionMode::Ctbr;
    assert_eq!(cfg.act_dim(), 4);
    cfg.action = ActionMode::Velocity;
    assert_eq!(cfg.act_dim(), 3);
    assert_eq!(EnvConfig::race(Airframe::racer()).obs_dim(), 18 + 4 + 6);
}

#[test]
fn rejects_bad_rates() {
    let mut cfg = EnvConfig::reach(Airframe::racer());
    cfg.control_hz = 7;
    assert!(BatchEnv::new(cfg, 1, 0, None).is_err());
}

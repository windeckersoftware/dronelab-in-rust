//! Many environments stepped in parallel with rayon.
//!
//! Each environment owns its random stream, so results do not depend on
//! how rayon schedules the work or how many threads it uses.

use std::sync::Arc;

use rayon::prelude::*;

use crate::env::{Env, EnvConfig, StepInfo};

pub struct BatchEnv {
    cfg: Arc<EnvConfig>,
    envs: Vec<Env>,
    /// Latest observations, `num_envs * obs_dim`, row-major.
    pub obs: Vec<f32>,
    /// For environments that finished this step, the final observation
    /// before the automatic reset (other rows are stale).
    pub terminal_obs: Vec<f32>,
    pub infos: Vec<StepInfo>,
    pool: Option<rayon::ThreadPool>,
}

impl BatchEnv {
    /// `num_envs` environments seeded `seed, seed + 1, ...`. `threads`
    /// limits parallelism (`None` uses rayon's global pool, one thread per core).
    pub fn new(
        cfg: EnvConfig,
        num_envs: usize,
        seed: u64,
        threads: Option<usize>,
    ) -> Result<Self, String> {
        cfg.validate()?;
        let cfg = Arc::new(cfg);
        let pool = threads
            .map(|n| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(n)
                    .build()
                    .map_err(|e| e.to_string())
            })
            .transpose()?;
        let od = cfg.obs_dim();
        let mut batch = BatchEnv {
            envs: (0..num_envs)
                .map(|i| Env::new(Arc::clone(&cfg), seed.wrapping_add(i as u64)))
                .collect(),
            obs: vec![0.0; num_envs * od],
            terminal_obs: vec![0.0; num_envs * od],
            infos: vec![StepInfo::default(); num_envs],
            pool,
            cfg,
        };
        batch.observe_all();
        Ok(batch)
    }

    pub fn config(&self) -> &EnvConfig {
        &self.cfg
    }

    pub fn num_envs(&self) -> usize {
        self.envs.len()
    }

    pub fn envs(&self) -> &[Env] {
        &self.envs
    }

    pub fn envs_mut(&mut self) -> &mut [Env] {
        &mut self.envs
    }

    fn run<F: FnOnce() + Send>(pool: &Option<rayon::ThreadPool>, f: F) {
        match pool {
            Some(p) => p.install(f),
            None => f(),
        }
    }

    /// Below this many environments, stepping inline beats handing work to
    /// other threads.
    const PARALLEL_MIN: usize = 8;

    fn observe_all(&mut self) {
        let od = self.cfg.obs_dim();
        let BatchEnv {
            envs, obs, pool, ..
        } = self;
        Self::run(pool, || {
            envs.par_iter_mut()
                .zip(obs.par_chunks_mut(od))
                .for_each(|(e, o)| e.observe(o));
        });
    }

    /// Resets every environment: env `i` gets `seed + i`, or continues its
    /// own random stream if `seed` is `None`.
    pub fn reset(&mut self, seed: Option<u64>) {
        let seeds: Vec<Option<u64>> = (0..self.envs.len())
            .map(|i| seed.map(|s| s.wrapping_add(i as u64)))
            .collect();
        self.reset_each(&seeds);
    }

    /// Resets env `i` with `seeds[i]`.
    pub fn reset_each(&mut self, seeds: &[Option<u64>]) {
        assert_eq!(seeds.len(), self.envs.len());
        let BatchEnv { envs, pool, .. } = self;
        Self::run(pool, || {
            envs.par_iter_mut()
                .zip(seeds.par_iter())
                .for_each(|(e, &s)| e.reset(s));
        });
        self.observe_all();
    }

    /// Steps every environment with its row of `actions`
    /// (`num_envs * act_dim`). Finished environments reset automatically:
    /// their final observation goes to `terminal_obs` and `obs` holds the
    /// first observation of the next episode, as vectorized RL expects.
    pub fn step(&mut self, actions: &[f32]) {
        let (ad, od) = (self.cfg.act_dim(), self.cfg.obs_dim());
        assert_eq!(
            actions.len(),
            self.envs.len() * ad,
            "actions must be num_envs * act_dim"
        );
        let BatchEnv {
            envs,
            obs,
            terminal_obs,
            infos,
            pool,
            ..
        } = self;
        type Row<'a> = (
            ((&'a mut Env, &'a [f32]), &'a mut [f32]),
            (&'a mut [f32], &'a mut StepInfo),
        );
        let step_one = |(((env, a), o), (t, info)): Row<'_>| {
            *info = env.step(a, o);
            if info.done() {
                t.copy_from_slice(o);
                env.reset(None);
                env.observe(o);
            }
        };
        if envs.len() < Self::PARALLEL_MIN {
            envs.iter_mut()
                .zip(actions.chunks(ad))
                .zip(obs.chunks_mut(od))
                .zip(terminal_obs.chunks_mut(od).zip(infos.iter_mut()))
                .for_each(step_one);
            return;
        }
        Self::run(pool, || {
            envs.par_iter_mut()
                .zip(actions.par_chunks(ad))
                .zip(obs.par_chunks_mut(od))
                .zip(terminal_obs.par_chunks_mut(od).zip(infos.par_iter_mut()))
                .with_min_len(4)
                .for_each(step_one);
        });
    }

    /// The built-in PID controller's actions for every environment.
    pub fn expert_actions(&mut self, out: &mut [f32]) -> Option<()> {
        let ad = self.cfg.act_dim();
        assert_eq!(out.len(), self.envs.len() * ad);
        let BatchEnv { envs, pool, .. } = self;
        let mut ok = true;
        Self::run(pool, || {
            ok = envs
                .par_iter_mut()
                .zip(out.par_chunks_mut(ad))
                .all(|(e, o)| e.expert_action(o).is_some());
        });
        ok.then_some(())
    }
}

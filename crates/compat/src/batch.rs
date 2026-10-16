//! Many aviaries stepped in parallel, with the automatic reset a
//! vectorised RL environment expects.

use rayon::prelude::*;

use crate::aviary::{Aviary, AviaryConfig, AviaryError, StepResult};

pub struct AviaryBatch {
    pub aviaries: Vec<Aviary>,
    /// Latest observations, row-major `(num_envs, num_drones * obs_dim)`.
    pub obs: Vec<f64>,
    /// Final observation of episodes that ended in the last step.
    pub terminal_obs: Vec<f64>,
    pub rewards: Vec<f64>,
    pub terminated: Vec<bool>,
    pub truncated: Vec<bool>,
}

impl AviaryBatch {
    pub fn new(cfg: AviaryConfig, num_envs: usize) -> Result<Self, AviaryError> {
        let proto = Aviary::new(cfg)?;
        let width = proto.num_drones() * proto.obs_dim();
        let mut aviaries = vec![proto; num_envs];
        let mut obs = vec![0.0; num_envs * width];
        for (a, o) in aviaries.iter_mut().zip(obs.chunks_mut(width)) {
            o.copy_from_slice(&a.reset());
        }
        Ok(AviaryBatch {
            aviaries,
            terminal_obs: obs.clone(),
            obs,
            rewards: vec![0.0; num_envs],
            terminated: vec![false; num_envs],
            truncated: vec![false; num_envs],
        })
    }

    pub fn num_envs(&self) -> usize {
        self.aviaries.len()
    }

    /// Per-env observation width (`num_drones * obs_dim`).
    pub fn obs_width(&self) -> usize {
        self.aviaries
            .first()
            .map_or(0, |a| a.num_drones() * a.obs_dim())
    }

    /// Per-env action width (`num_drones * act_dim`).
    pub fn act_width(&self) -> usize {
        self.aviaries
            .first()
            .map_or(0, |a| a.num_drones() * a.act_dim())
    }

    pub fn reset(&mut self) {
        let w = self.obs_width();
        self.aviaries
            .par_iter_mut()
            .zip(self.obs.par_chunks_mut(w))
            .for_each(|(a, o)| o.copy_from_slice(&a.reset()));
    }

    /// Steps every aviary; finished episodes reset automatically.
    pub fn step(&mut self, actions: &[f64]) {
        let (w, aw) = (self.obs_width(), self.act_width());
        assert_eq!(
            actions.len(),
            self.num_envs() * aw,
            "actions must be num_envs * num_drones * act_dim"
        );
        let results: Vec<StepResult> = self
            .aviaries
            .par_iter_mut()
            .zip(actions.par_chunks(aw))
            .zip(
                self.obs
                    .par_chunks_mut(w)
                    .zip(self.terminal_obs.par_chunks_mut(w)),
            )
            .with_min_len(4)
            .map(|((a, act), (o, t))| {
                let r = a.step(act);
                if r.terminated || r.truncated {
                    t.copy_from_slice(&r.obs);
                    o.copy_from_slice(&a.reset());
                } else {
                    o.copy_from_slice(&r.obs);
                }
                StepResult {
                    obs: Vec::new(),
                    ..r
                }
            })
            .collect();
        for (i, r) in results.into_iter().enumerate() {
            self.rewards[i] = r.reward;
            self.terminated[i] = r.terminated;
            self.truncated[i] = r.truncated;
        }
    }
}

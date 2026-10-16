"""Stable-Baselines3 ``VecEnv`` that steps the whole batch in Rust.

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecMonitor
    from dronelab.vec_env import DronelabVecEnv

    env = VecMonitor(DronelabVecEnv(num_envs=256, airframe="evtol_single_seat"))
    PPO("MlpPolicy", env).learn(1_000_000)
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices

from ._native import BatchSim


class DronelabVecEnv(VecEnv):
    """``num_envs`` environments in one Rust batch. Keyword arguments go to
    ``BatchSim`` (see ``dronelab.DronelabEnv``)."""

    def __init__(self, num_envs: int = 64, seed: int = 0, **kwargs: Any):
        self.sim = BatchSim(num_envs=num_envs, seed=seed, **kwargs)
        self.render_mode = None
        obs_space = spaces.Box(-np.inf, np.inf, (self.sim.obs_dim,), np.float32)
        act_space = spaces.Box(-1.0, 1.0, (self.sim.act_dim,), np.float32)
        super().__init__(num_envs, obs_space, act_space)
        self._actions = np.zeros((num_envs, self.sim.act_dim), np.float32)

    def reset(self):
        if any(s is not None for s in self._seeds):
            obs = self.sim.reset(seeds=list(self._seeds))
        else:
            obs = self.sim.reset()
        self._reset_seeds()
        self._reset_options()
        return obs

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = np.ascontiguousarray(actions, dtype=np.float32).reshape(self.num_envs, -1)

    def step_wait(self):
        obs, reward, terminated, truncated, info = self.sim.step(self._actions)
        dones = terminated | truncated
        infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]
        for i in np.flatnonzero(dones):
            infos[i]["terminal_observation"] = info["terminal_obs"][i]
            # SB3 bootstraps the value of time-limited, not failed, episodes.
            infos[i]["TimeLimit.truncated"] = bool(truncated[i] and not terminated[i])
            infos[i]["crashed"] = bool(info["crashed"][i])
        for i in np.flatnonzero(~np.isnan(info["lap_time"])):
            infos[i]["lap_time"] = float(info["lap_time"][i])
        return obs, reward, dones, infos

    def expert_actions(self) -> np.ndarray:
        return self.sim.expert_actions()

    def close(self) -> None:
        pass

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list[Any]:
        target = self.sim if hasattr(self.sim, attr_name) else self
        value = getattr(target, attr_name)
        return [value for _ in self._get_indices(indices)]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        setattr(self, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> list[Any]:
        result = getattr(self.sim, method_name)(*method_args, **method_kwargs)
        return [result for _ in self._get_indices(indices)]

    def env_is_wrapped(self, wrapper_class, indices: VecEnvIndices = None) -> list[bool]:
        return [False for _ in self._get_indices(indices)]

    def seed(self, seed: int | None = None) -> Sequence[None | int]:
        return super().seed(seed)

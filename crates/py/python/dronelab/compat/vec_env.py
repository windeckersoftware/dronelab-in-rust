"""Stable-Baselines3 VecEnv running many upstream-compatible aviaries in Rust.

    env = CompatVecEnv("hover", num_envs=64, act=ActionType.RPM, physics=Physics.PYB)
"""

from typing import Any

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices

from .. import _native
from .envs import HoverAviary, MultiHoverAviary
from .utils.enums import ActionType, DroneModel, ObservationType, Physics, value_of


class CompatVecEnv(VecEnv):
    """``num_envs`` copies of HoverAviary (``env="hover"``) or
    MultiHoverAviary (``env="multihover"``), stepped in parallel. Episodes
    reset automatically, as SB3 expects. Observations and actions are
    flattened over drones (shape ``(num_envs, num_drones * dim)``)."""

    def __init__(self, env: str = "hover", num_envs: int = 8, drone_model=DroneModel.CF2X, num_drones: int = None,
                 neighbourhood_radius: float = np.inf, initial_xyzs=None, initial_rpys=None, physics=Physics.PYB,
                 pyb_freq: int = 240, ctrl_freq: int = 30, obs=ObservationType.KIN, act=ActionType.RPM):
        cls = {"hover": HoverAviary, "multihover": MultiHoverAviary}[env]
        n = 1 if env == "hover" else (num_drones or 2)
        kw = dict(drone_model=drone_model, initial_xyzs=initial_xyzs, initial_rpys=initial_rpys, physics=physics,
                  pyb_freq=pyb_freq, ctrl_freq=ctrl_freq, obs=obs, act=act)
        if env == "multihover":
            kw.update(num_drones=n, neighbourhood_radius=neighbourhood_radius)
        template = cls(**kw)  # validates arguments, defines the spaces
        self.render_mode = None
        obs_space = spaces.Box(low=template.observation_space.low.reshape(-1), high=template.observation_space.high.reshape(-1), dtype=np.float32)
        act_space = spaces.Box(low=template.action_space.low.reshape(-1), high=template.action_space.high.reshape(-1), dtype=np.float32)
        xyz = None if initial_xyzs is None else np.ascontiguousarray(np.asarray(initial_xyzs, np.float64))
        rpy = None if initial_rpys is None else np.ascontiguousarray(np.asarray(initial_rpys, np.float64))
        self.batch = _native.CompatBatch(num_envs, env, value_of(act), value_of(drone_model), n, float(neighbourhood_radius),
                                         xyz, rpy, value_of(physics), pyb_freq, ctrl_freq)
        super().__init__(num_envs, obs_space, act_space)
        self._actions = np.zeros((num_envs, act_space.shape[0]))

    def reset(self):
        self._reset_seeds()
        self._reset_options()
        return self.batch.reset()

    def step_async(self, actions):
        self._actions = np.ascontiguousarray(np.asarray(actions, dtype=np.float64).reshape(self.num_envs, -1))

    def step_wait(self):
        obs, rewards, terminated, truncated, terminal = self.batch.step(self._actions)
        dones = terminated | truncated
        infos: list[dict[str, Any]] = [{"answer": 42} for _ in range(self.num_envs)]
        for i in np.flatnonzero(dones):
            infos[i]["terminal_observation"] = terminal[i]
            infos[i]["TimeLimit.truncated"] = bool(truncated[i] and not terminated[i])
        return obs, rewards, dones, infos

    def close(self):
        pass

    def get_attr(self, attr_name, indices: VecEnvIndices = None):
        return [getattr(self, attr_name) for _ in self._get_indices(indices)]

    def set_attr(self, attr_name, value, indices: VecEnvIndices = None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *args, indices: VecEnvIndices = None, **kwargs):
        raise NotImplementedError

    def env_is_wrapped(self, wrapper_class, indices: VecEnvIndices = None):
        return [False for _ in self._get_indices(indices)]

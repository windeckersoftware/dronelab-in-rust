"""Gymnasium environment backed by the Rust simulator."""

from __future__ import annotations

from typing import Any

import json

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ._native import BatchSim


class DronelabEnv(gym.Env):
    """One vehicle, one task. Keyword arguments go to ``BatchSim``:
    ``airframe``, ``task`` ("reach" or "race"), ``action`` ("rotor_thrust",
    "ctbr", "velocity"), ``physics_hz``, ``control_hz``, ``episode_seconds``,
    ``realism`` ("full", "ideal" or a dict of flags), ``randomize`` (bool or
    dict of ranges), ``track_json``.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: str | None = None, **kwargs: Any):
        if render_mode not in (None, "rgb_array"):
            raise ValueError(f"render_mode must be None or 'rgb_array', not {render_mode!r}")
        self.render_mode = render_mode
        self.sim = BatchSim(num_envs=1, **kwargs)
        self.metadata = {**self.metadata, "render_fps": self.sim.control_hz}
        track = self.sim.track_json()
        self._gates = json.loads(track)["gates"] if track else []
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.sim.obs_dim,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (self.sim.act_dim,), np.float32)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        obs = self.sim.reset(seed=seed)
        return obs[0], {}

    def step(self, action):
        a = np.asarray(action, dtype=np.float32).reshape(1, -1)
        obs, reward, terminated, truncated, info = self.sim.step(a)
        done = bool(terminated[0] or truncated[0])
        # The batch resets finished episodes itself; Gymnasium expects the
        # final observation here and a reset() call from the user.
        out = info["terminal_obs"][0] if done else obs[0]
        step_info = {"crashed": bool(info["crashed"][0]), "gate_passed": bool(info["gate_passed"][0])}
        if not np.isnan(info["lap_time"][0]):
            step_info["lap_time"] = float(info["lap_time"][0])
        return out, float(reward[0]), bool(terminated[0]), bool(truncated[0]), step_info

    def render(self, size: int = 256) -> np.ndarray | None:
        """Top-down view: vehicle (blue), target (red), gates (grey)."""
        if self.render_mode != "rgb_array":
            return None
        st = self.sim.state()
        pos, target = st["pos"][0], st["target"][0]
        points = [pos[:2], target[:2]] + [np.array(g["center"][:2]) for g in self._gates]
        lo = np.min(points, axis=0) - 2 * self.sim.radius
        hi = np.max(points, axis=0) + 2 * self.sim.radius
        span = float(max(hi - lo)) or 1.0
        img = np.full((size, size, 3), 255, np.uint8)

        def px(p):
            q = (np.asarray(p[:2]) - lo) / span * (size - 1)
            return int(q[0]), int(size - 1 - q[1])

        def dot(p, r, color):
            x, y = px(p)
            img[max(y - r, 0) : y + r + 1, max(x - r, 0) : x + r + 1] = color

        for g in self._gates:
            c, half = np.array(g["center"]), g["width"] / 2
            side = np.array([-np.sin(g["yaw"]), np.cos(g["yaw"])])
            for t in np.linspace(-1, 1, 20):
                dot(c[:2] + side * half * t, 1, (150, 150, 150))
        dot(target, 3, (220, 40, 40))
        dot(pos, max(2, int(self.sim.radius / span * size)), (40, 90, 220))
        return img

    def expert_action(self) -> np.ndarray:
        """What the built-in PID controller would do now."""
        return self.sim.expert_actions()[0]

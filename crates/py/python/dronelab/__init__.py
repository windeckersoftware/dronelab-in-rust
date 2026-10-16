"""dronelab: fast, deterministic multirotor and eVTOL simulation for RL.

- ``BatchSim``: thousands of environments stepped in parallel in Rust.
- ``DronelabEnv``: a single Gymnasium environment.
- ``dronelab.vec_env.DronelabVecEnv``: a Stable-Baselines3 ``VecEnv`` that
  runs the whole batch in Rust (needs ``stable-baselines3``).
"""

from ._native import BatchSim, Leaderboard, __version__, airframes, default_track_json
from .envs import DronelabEnv

__all__ = ["BatchSim", "DronelabEnv", "Leaderboard", "airframes", "default_track_json", "__version__"]


def _register():
    from gymnasium.envs.registration import register, registry

    for task in ("reach", "race"):
        for airframe in airframes():
            env_id = f"dronelab/{task.capitalize()}-{airframe}-v0"
            if env_id not in registry:
                register(id=env_id, entry_point="dronelab.envs:DronelabEnv", kwargs={"airframe": airframe, "task": task})


_register()

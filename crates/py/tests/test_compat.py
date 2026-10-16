"""dronelab.compat without upstream installed: API shape, Gymnasium, SB3."""

import subprocess
import sys

import numpy as np
import pytest

from dronelab import compat
from dronelab.compat.envs.HoverAviary import HoverAviary
from dronelab.compat.utils.enums import ActionType, Physics


def test_observation_dtype_matches_upstream():
    # Upstream returns float64 observations from a float32 Box, so
    # Gymnasium's env checker rejects it; the port keeps that (checked
    # against v2.2.0). SB3 casts the observations and trains fine.
    env = HoverAviary(physics=Physics.DYN, act=ActionType.RPM)
    obs, info = env.reset(seed=0)
    assert obs.dtype == np.float64 and env.observation_space.dtype == np.float32
    assert obs.shape == env.observation_space.shape == (1, 72)
    assert info == {"answer": 42}


def test_unsupported_features_fail_loudly():
    with pytest.raises(NotImplementedError):
        HoverAviary(gui=True)
    with pytest.raises(NotImplementedError):
        HoverAviary(obs=compat.ObservationType.RGB)
    with pytest.raises(ValueError):
        compat.DSLPIDControl(compat.DroneModel.RACE)
    with pytest.warns(UserWarning, match="obstacles"):
        compat.CtrlAviary(obstacles=True)


def test_vec_env_matches_single_envs():
    from dronelab.compat.vec_env import CompatVecEnv

    vec = CompatVecEnv("multihover", num_envs=3, physics=Physics.DYN, act=ActionType.RPM)
    single = compat.MultiHoverAviary(physics=Physics.DYN, act=ActionType.RPM)
    obs = vec.reset()
    o, _ = single.reset()
    np.testing.assert_array_equal(obs[0], o.reshape(-1))
    a = np.random.default_rng(0).uniform(-1, 1, (3, 8))
    obs, rew, done, info = vec.step(a)
    o, r, *_ = single.step(a[0].reshape(2, 4))
    np.testing.assert_array_equal(obs[0], o.reshape(-1))
    assert rew[0] == r


def test_sb3_ppo_trains_on_compat_vec_env():
    from stable_baselines3 import PPO

    from dronelab.compat.vec_env import CompatVecEnv

    env = CompatVecEnv("hover", num_envs=16, physics=Physics.PYB, act=ActionType.ONE_D_RPM)
    PPO("MlpPolicy", env, n_steps=32, batch_size=256, n_epochs=1, device="cpu", seed=0).learn(1024)


def test_install_makes_upstream_imports_resolve_here():
    code = (
        "import dronelab.compat as c; c.install()\n"
        "from gym_pybullet_drones.envs.HoverAviary import HoverAviary\n"
        "from gym_pybullet_drones.utils.enums import ActionType\n"
        "from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl\n"
        "import sys\n"
        "assert HoverAviary is sys.modules['dronelab.compat.envs.HoverAviary'].HoverAviary\n"
        "env = HoverAviary(act=ActionType.PID); env.reset(); env.step([[0, 0, 1]])\n"
        "print('ok')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.stdout.strip().endswith("ok"), out.stderr

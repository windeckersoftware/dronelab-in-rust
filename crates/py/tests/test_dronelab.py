import json

import gymnasium as gym
import numpy as np
import pytest

import dronelab
from dronelab import BatchSim, DronelabEnv, Leaderboard


@pytest.mark.parametrize("task", ["reach", "race"])
@pytest.mark.parametrize("action", ["rotor_thrust", "ctbr", "velocity"])
def test_gymnasium_env_checker(task, action):
    from gymnasium.utils.env_checker import check_env

    check_env(DronelabEnv(airframe="evtol_single_seat", task=task, action=action), skip_render_check=True)


def test_sb3_env_checker():
    from stable_baselines3.common.env_checker import check_env

    check_env(DronelabEnv(airframe="evtol_two_seat"), warn=True)


def test_registered_ids():
    env = gym.make("dronelab/Race-racer-v0")
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape


def test_same_seed_same_flight():
    def run(seed):
        sim = BatchSim(num_envs=32, airframe="crazyflie_x", seed=seed)
        rng = np.random.default_rng(0)
        out = [sim.reset(seed=seed)]
        for _ in range(100):
            out.append(sim.step(rng.uniform(-0.3, 0.3, (32, sim.act_dim)).astype(np.float32))[0])
        return np.stack(out)

    a, b = run(4), run(4)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, run(5))


def test_gym_episode_end_returns_final_observation():
    env = DronelabEnv(airframe="racer", episode_seconds=0.5)
    obs, _ = env.reset(seed=3)
    for _ in range(15):
        obs, _, term, trunc, _ = env.step(env.expert_action())
    assert trunc and not term
    # Final observation continues the episode instead of starting a new one.
    first, _ = env.reset(seed=3)
    assert not np.allclose(obs, first)


def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        BatchSim(airframe="jetpack")
    with pytest.raises(KeyError):
        BatchSim(realism={"antigravity": True})
    sim = BatchSim(num_envs=2)
    with pytest.raises(ValueError):
        sim.step(np.zeros((3, sim.act_dim), np.float32))


def test_realism_and_randomization_options():
    ideal = BatchSim(num_envs=8, realism="ideal", randomize=False)
    st = ideal.state()
    assert np.all(st["battery_soc"] == 1.0) and np.all(st["wind"] == 0.0)
    heavy = BatchSim(num_envs=8, randomize={"mass_scale": [1.3, 1.3], "wind_speed": [5.0, 5.0]})
    st = heavy.state()
    assert np.allclose(st["mass"], 1.3 * 200.0)
    assert np.allclose(np.linalg.norm(st["wind"][:, :2], axis=1), 5.0, atol=1e-4)


def test_custom_track_and_leaderboard(tmp_path):
    track = json.loads(dronelab.default_track_json("racer"))
    track["name"] = "wide"
    for g in track["gates"]:
        g["width"] *= 1.5
    sim = BatchSim(num_envs=1, airframe="racer", task="race", action="velocity", track_json=json.dumps(track), episode_seconds=40)
    assert json.loads(sim.track_json())["name"] == "wide"
    sim.reset(seed=1)
    laps = []
    for _ in range(sim.episode_steps):
        _, _, term, _, info = sim.step(sim.expert_actions())
        assert not term[0]
        if not np.isnan(info["lap_time"][0]):
            laps.append(float(info["lap_time"][0]))
    assert len(laps) >= 2

    board = Leaderboard(str(tmp_path / "board.json"), capacity=3)
    for t in laps:
        board.submit("pid", "racer", "wide", t, 1)
    board.submit("slow", "racer", "wide", 99.0, 2)
    top = Leaderboard(str(tmp_path / "board.json")).top("wide", "racer")
    assert [r["lap_time"] for r in top] == sorted(r["lap_time"] for r in top)
    assert top[0]["lap_time"] == min(laps)


def test_sb3_ppo_trains_on_rust_vec_env():
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecMonitor

    from dronelab.vec_env import DronelabVecEnv

    env = VecMonitor(DronelabVecEnv(num_envs=64, airframe="evtol_single_seat", episode_seconds=2.0))
    env.seed(123)
    model = PPO("MlpPolicy", env, n_steps=64, batch_size=1024, n_epochs=2, device="cpu", seed=0)
    model.learn(total_timesteps=64 * 64 * 2)
    assert len(model.ep_info_buffer) > 0


def test_sb3_ppo_trains_on_plain_gymnasium_id():
    from stable_baselines3 import PPO

    model = PPO("MlpPolicy", "dronelab/Reach-crazyflie_x-v0", n_steps=128, batch_size=64, n_epochs=1, device="cpu", seed=0)
    model.learn(total_timesteps=256)


def test_rgb_array_render():
    env = DronelabEnv(airframe="evtol_single_seat", task="race", render_mode="rgb_array")
    env.reset(seed=0)
    img = env.render()
    assert img.shape == (256, 256, 3) and img.dtype == np.uint8
    assert (img != 255).any()

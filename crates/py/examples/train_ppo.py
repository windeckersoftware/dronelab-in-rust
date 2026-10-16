"""Train PPO (Stable-Baselines3) on a dronelab task, with the batch in Rust.

    python crates/py/examples/train_ppo.py --airframe evtol_single_seat --steps 2000000
"""

import argparse
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor

from dronelab.vec_env import DronelabVecEnv


def evaluate(model, kwargs, episodes=64):
    """Mean return and crash rate over `episodes` fresh episodes."""
    env = DronelabVecEnv(num_envs=episodes, **kwargs)
    env.seed(10_000)  # the same episodes for every evaluation
    obs = env.reset()
    total, done_any, crashed = np.zeros(episodes), np.zeros(episodes, bool), np.zeros(episodes, bool)
    while not done_any.all():
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = env.step(action)
        total += reward * ~done_any
        for i in np.flatnonzero(dones & ~done_any):
            crashed[i] = infos[i].get("crashed", False)
        done_any |= dones
    return total.mean(), crashed.mean()


def _reference_returns(env):
    """Returns of doing nothing and of the built-in PID expert."""
    out = []
    for use_expert in (False, True):
        env.seed(10_000)
        obs = env.reset()
        total, done_any = np.zeros(env.num_envs), np.zeros(env.num_envs, bool)
        while not done_any.all():
            a = env.expert_actions() if use_expert else np.zeros((env.num_envs, env.sim.act_dim), np.float32)
            obs, reward, dones, _ = env.step(a)
            total += reward * ~done_any
            done_any |= dones
        out.append(total.mean())
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--airframe", default="evtol_single_seat")
    p.add_argument("--task", default="reach")
    p.add_argument("--action", default="velocity")
    p.add_argument("--envs", type=int, default=256)
    p.add_argument("--steps", type=int, default=2_000_000)
    args = p.parse_args()

    kwargs = dict(airframe=args.airframe, task=args.task, action=args.action)
    env = VecMonitor(DronelabVecEnv(num_envs=args.envs, seed=0, **kwargs))
    # Long rollouts and modest initial exploration noise: per-step action
    # noise hardly moves a heavy vehicle, so short rollouts give PPO mostly
    # noise to learn from.
    model = PPO(
        "MlpPolicy", env, n_steps=256, batch_size=8192, n_epochs=10, device="cpu", seed=0,
        policy_kwargs=dict(log_std_init=-1.0),
    )

    baseline = DronelabVecEnv(num_envs=64, **kwargs)
    hover, expert = _reference_returns(baseline)
    print(f"reference returns: hold still {hover:.1f}, PID expert {expert:.1f}")
    print(f"before training: return {evaluate(model, kwargs)[0]:.1f}")
    start = time.time()
    chunk = args.steps // 5
    for k in range(5):
        model.learn(chunk, reset_num_timesteps=False)
        ret, crash = evaluate(model, kwargs)
        print(f"{(k + 1) * chunk:>9} steps  {time.time() - start:6.0f} s  return {ret:7.1f}  crash rate {crash:.0%}")
    model.save(f"ppo_{args.airframe}_{args.task}")


if __name__ == "__main__":
    main()

"""Steps per second: gym-pybullet-drones (PyBullet) against dronelab.

Both simulators run a Crazyflie (CF2X / crazyflie_x) hover task at 240 Hz
physics and 30 Hz control (8 physics steps per env step), with automatic
resets, the way an RL rollout would run them. Two action workloads:

  random  uniform random actions; episodes end (tilt, crash) after tens of
          steps, so reset cost is part of the number.
  hover   the zero action (open-loop hover thrust); HoverAviary episodes
          run to the 8 s limit, so its number is close to pure stepping.
          dronelab's still end early (drift), but its resets are cheap.

The last column gives episodes ended per 1000 env steps for both.

    bench/.venv-pybullet/bin/python bench/pybullet_vs_dronelab.py

"env steps/s" counts env.step() calls summed over all environments;
"physics steps/s" is that times 8. Each row is the median of --repeats timed
runs of --seconds each; env construction and a warm-up are not timed.

Random actions are matched in size. HoverAviary's RPM action maps a in
[-1, 1] to hover_rpm * (1 + 0.05 a), so +-10% of hover thrust; dronelab's
rotor_thrust action maps a to hover_thrust * (1 + a), so it gets a * 0.1.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import warnings
from functools import partial
from importlib.metadata import version

import numpy as np

PYB_HZ, CTRL_HZ = 240, 30
SUBSTEPS = PYB_HZ // CTRL_HZ
POOL = 64  # pre-generated random action batches, cycled


def timed(step, args) -> dict:
    """Calls step(i) -> (env steps, episodes ended) for args.warmup untimed
    seconds, then args.repeats runs of args.seconds. Returns the median env
    steps/s and the episodes ended per 1000 env steps."""
    i, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < args.warmup:
        step(i)
        i += 1
    rates, ends, total = [], 0, 0
    for _ in range(args.repeats):
        steps = 0
        t0 = time.perf_counter()
        while (dt := time.perf_counter() - t0) < args.seconds:
            n, e = step(i)
            steps += n
            ends += e
            i += 1
        rates.append(steps / dt)
        total += steps
    return dict(env_steps_per_s=statistics.median(rates), physics_steps_per_s=statistics.median(rates) * SUBSTEPS,
                episode_ends_per_1k=1000 * ends / total)


def action_pools(rng, shape) -> dict[str, np.ndarray]:
    return {
        "random": rng.uniform(-1, 1, (POOL, *shape)).astype(np.float32),
        "hover": np.zeros((POOL, *shape), np.float32),
    }


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


# ---------------------------------------------------------------- PyBullet


def pybullet_rows(args, rng) -> list[dict]:
    from gym_pybullet_drones.envs.HoverAviary import HoverAviary
    from gym_pybullet_drones.utils.enums import Physics
    from stable_baselines3.common.vec_env import SubprocVecEnv

    rows = []
    for physics in (Physics.PYB, Physics.DYN):
        make = partial(quiet, HoverAviary, physics=physics, pyb_freq=PYB_HZ, ctrl_freq=CTRL_HZ)

        env = make()
        res = {}
        for load, acts in action_pools(rng, (1, 4)).items():
            quiet(env.reset, seed=0)

            def step(i, env=env, acts=acts):
                _, _, term, trunc, _ = env.step(acts[i % POOL])
                if term or trunc:
                    quiet(env.reset)
                    return 1, 1
                return 1, 0

            res[load] = timed(step, args)
        env.close()
        rows.append(row("gym-pybullet-drones", physics.name, "HoverAviary", 1, "1 process", res))

        for n in args.procs:
            venv = SubprocVecEnv([make] * n)
            res = {}
            for load, acts in action_pools(rng, (n, 1, 4)).items():
                venv.seed(0)
                venv.reset()

                def step(i, venv=venv, acts=acts, n=n):
                    _, _, dones, _ = venv.step(acts[i % POOL])
                    return n, int(dones.sum())

                res[load] = timed(step, args)
            venv.close()
            rows.append(row("gym-pybullet-drones", physics.name, "SB3 SubprocVecEnv", n, f"{n} process" + "es" * (n > 1), res))
    return rows


# ---------------------------------------------------------------- dronelab


def dronelab_rows(args, rng) -> list[dict]:
    from dronelab import BatchSim
    from dronelab.vec_env import DronelabVecEnv

    workers = f"{os.cpu_count()} threads"
    rows = []
    for realism in ("ideal", "full"):
        # ideal: no randomization, like HoverAviary; full: the airframe's defaults.
        kw = dict(
            airframe="crazyflie_x", task="hover", realism=realism, randomize=realism == "full",
            physics_hz=PYB_HZ, control_hz=CTRL_HZ, episode_seconds=8.0,
        )
        for n in args.envs:
            # rotor_thrust: a -> hover_thrust * (1 + a); scale to HoverAviary's +-10% thrust.
            pools = {load: 0.1 * acts for load, acts in action_pools(rng, (n, 4)).items()}

            sim = BatchSim(num_envs=n, seed=0, **kw)
            res = {}
            for load, acts in pools.items():
                sim.reset(seed=0)

                def step(i, sim=sim, acts=acts, n=n):
                    _, _, term, trunc, _ = sim.step(acts[i % POOL])
                    return n, int(np.count_nonzero(term | trunc))

                res[load] = timed(step, args)
            rows.append(row("dronelab", realism, "BatchSim", n, workers, res))

            venv = DronelabVecEnv(num_envs=n, seed=0, **kw)
            res = {}
            for load, acts in pools.items():
                venv.seed(0)
                venv.reset()

                def step(i, venv=venv, acts=acts, n=n):
                    _, _, dones, _ = venv.step(acts[i % POOL])
                    return n, int(np.count_nonzero(dones))

                res[load] = timed(step, args)
            rows.append(row("dronelab", realism, "SB3 DronelabVecEnv", n, workers, res))
    return rows


# ---------------------------------------------------------------- report


def row(sim, physics, api, envs, workers, res) -> dict:
    return dict(simulator=sim, physics=physics, api=api, envs=envs, workers=workers, **res)


def fmt(x: float) -> str:
    if x >= 1e6:
        return f"{x / 1e6:.2f} M"
    if x >= 1e3:
        return f"{x / 1e3:.1f} k"
    return f"{x:.0f}"


def machine() -> dict:
    def sysctl(key):
        try:
            return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True).stdout.strip()
        except OSError:
            return ""

    pkgs = {}
    for p in ("pybullet", "gym-pybullet-drones", "dronelab", "stable-baselines3", "gymnasium", "numpy"):
        try:
            pkgs[p] = version(p)
        except Exception:
            pass
    cpu = sysctl("machdep.cpu.brand_string") or platform.processor()
    perf, eff = sysctl("hw.perflevel0.physicalcpu"), sysctl("hw.perflevel1.physicalcpu")
    cores = f"{os.cpu_count()} cores" + (f" ({perf} performance + {eff} efficiency)" if perf and eff else "")
    return dict(cpu=cpu, cores=cores, os=f"{platform.system()} {platform.mac_ver()[0] or platform.release()}",
                python=platform.python_version(), packages=pkgs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=3.0, help="length of each timed run")
    ap.add_argument("--repeats", type=int, default=3, help="timed runs per row (median reported)")
    ap.add_argument("--warmup", type=float, default=0.5, help="untimed seconds before the timed runs")
    ap.add_argument("--procs", type=int, nargs="+", default=[1, 8, os.cpu_count()], help="SubprocVecEnv sizes")
    ap.add_argument("--envs", type=int, nargs="+", default=[1, 1024, 8192], help="dronelab batch sizes")
    ap.add_argument("--only", choices=["pybullet", "dronelab"], help="run one simulator only")
    ap.add_argument("--json", help="also write the results here")
    args = ap.parse_args()
    args.procs = sorted(set(args.procs))
    # gymnasium warns about float64 Box bounds in every HoverAviary, workers included.
    os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"
    warnings.simplefilter("ignore", UserWarning)

    info = machine()
    print(f"{info['cpu']}, {info['cores']}, {info['os']}, Python {info['python']}", file=sys.stderr)
    print(", ".join(f"{k} {v}" for k, v in info["packages"].items()), file=sys.stderr)

    rng = np.random.default_rng(0)
    rows = []
    for name, fn in (("pybullet", pybullet_rows), ("dronelab", dronelab_rows)):
        if args.only in (None, name):
            print(f"running {name} ...", file=sys.stderr)
            rows += fn(args, rng)

    print("| simulator | physics | API | envs | workers | random: env steps/s | random: physics steps/s "
          "| hover: env steps/s | hover: physics steps/s | episode ends / 1k steps (random, hover) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        rnd, hov = r["random"], r["hover"]
        print(f"| {r['simulator']} | {r['physics']} | {r['api']} | {r['envs']} | {r['workers']} "
              f"| {fmt(rnd['env_steps_per_s'])} | {fmt(rnd['physics_steps_per_s'])} "
              f"| {fmt(hov['env_steps_per_s'])} | {fmt(hov['physics_steps_per_s'])} "
              f"| {rnd['episode_ends_per_1k']:.0f}, {hov['episode_ends_per_1k']:.0f} |")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(machine=info, args=vars(args), rows=rows), f, indent=2)


if __name__ == "__main__":
    main()

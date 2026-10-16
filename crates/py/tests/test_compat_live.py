"""dronelab.compat side by side with the real gym-pybullet-drones.

Needs gym-pybullet-drones (and PyBullet) installed next to dronelab; skipped
otherwise. In this repository:

    bench/.venv-pybullet/bin/python -m pytest crates/py/tests/test_compat_live.py
"""

import glob
import io
import contextlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

gpd = pytest.importorskip("gym_pybullet_drones")

from gym_pybullet_drones.control.CTBRControl import CTBRControl as UpCTBR  # noqa: E402
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl as UpDSL  # noqa: E402
from gym_pybullet_drones.control.MRACControl import MRACControl as UpMRAC  # noqa: E402
from gym_pybullet_drones.envs import CtrlAviary as UpCtrl  # noqa: E402
from gym_pybullet_drones.envs import HoverAviary as UpHover  # noqa: E402
from gym_pybullet_drones.envs import MultiHoverAviary as UpMulti  # noqa: E402
from gym_pybullet_drones.envs import VelocityAviary as UpVel  # noqa: E402
from gym_pybullet_drones.utils import enums as up_enums  # noqa: E402

from dronelab import compat  # noqa: E402

UPSTREAM_EXAMPLES = Path(gpd.__file__).parent / "examples"

PAIRS = {
    "ctrl": (UpCtrl, compat.CtrlAviary),
    "velocity": (UpVel, compat.VelocityAviary),
    "hover": (UpHover, compat.HoverAviary),
    "multihover": (UpMulti, compat.MultiHoverAviary),
}


def quiet():
    return contextlib.redirect_stdout(io.StringIO())


def both(name, **kw):
    """Builds the upstream and the compat environment with the same arguments."""
    up_cls, our_cls = PAIRS[name]
    up_kw = {k: (getattr(up_enums, type(v).__name__)(v.value) if hasattr(v, "value") else v) for k, v in kw.items()}
    with quiet():
        return up_cls(**up_kw), our_cls(**kw)


def close_to(ours, up, tol):
    ours, up = np.asarray(ours, float), np.asarray(up, float)
    assert ours.shape == up.shape
    # RL observations are float32-rounded upstream: allow one float32 step.
    slack = tol + np.abs(up) * np.finfo(np.float32).eps
    bad = np.abs(ours - up) > slack
    assert not bad.any(), f"max error {np.abs(ours - up).max():.3e} (tol {tol:.0e})"


@pytest.mark.parametrize("physics", [compat.Physics.DYN, compat.Physics.PYB, compat.Physics.PYB_GND_DRAG_DW])
@pytest.mark.parametrize("name,act", [
    ("hover", compat.ActionType.RPM), ("hover", compat.ActionType.PID), ("hover", compat.ActionType.VEL),
    ("hover", compat.ActionType.ONE_D_RPM), ("hover", compat.ActionType.ONE_D_PID),
    ("multihover", compat.ActionType.RPM), ("multihover", compat.ActionType.PID),
])
def test_rl_envs_step_like_upstream(name, act, physics):
    n = 1 if name == "hover" else 2
    start = np.array([[0.0, 0.0, 1.0], [0.4, 0.0, 1.2]])[:n]
    up, ours = both(name, physics=physics, act=act, initial_xyzs=start)
    assert ours.observation_space == up.observation_space
    assert ours.action_space == up.action_space
    for attr in ["M", "L", "KF", "KM", "GRAVITY", "HOVER_RPM", "MAX_RPM", "CTRL_TIMESTEP", "PYB_STEPS_PER_CTRL", "EPISODE_LEN_SEC"]:
        assert getattr(ours, attr) == getattr(up, attr), attr
    np.testing.assert_array_equal(ours.TARGET_POS, up.TARGET_POS)
    tol = 1e-9 if physics == compat.Physics.DYN else 1e-5
    with quiet():
        (o_up, i_up), (o_us, i_us) = up.reset(seed=1), ours.reset(seed=1)
    close_to(o_us, o_up, tol)
    assert i_us == i_up
    rng = np.random.default_rng(5)
    for _ in range(15):
        a = rng.uniform(-0.3, 0.3, size=up.action_space.shape)
        if act == compat.ActionType.PID:
            a = start + rng.normal(scale=0.05, size=a.shape)
        with quiet():
            r_up, r_us = up.step(a), ours.step(a)
        close_to(r_us[0], r_up[0], tol)
        assert abs(r_us[1] - r_up[1]) <= 1e3 * tol
        assert r_us[2:4] == r_up[2:4]
        np.testing.assert_allclose(ours.pos, up.pos, atol=tol)
    up.close()


@pytest.mark.parametrize("physics", [compat.Physics.DYN, compat.Physics.PYB_DW])
def test_ctrl_and_velocity_aviaries_step_like_upstream(physics):
    start = np.array([[0.0, 0.0, 0.8], [0.05, 0.0, 1.2]])
    for name in ["ctrl", "velocity"]:
        up, ours = both(name, num_drones=2, physics=physics, initial_xyzs=start, ctrl_freq=48)
        assert ours.action_space == up.action_space and ours.observation_space == up.observation_space
        tol = 1e-9 if physics == compat.Physics.DYN else 1e-6
        rng = np.random.default_rng(2)
        for _ in range(20):
            if name == "ctrl":
                a = up.HOVER_RPM * (1 + rng.uniform(-0.03, 0.03, (2, 4)))
            else:
                a = np.hstack([rng.uniform(-1, 1, (2, 3)), rng.uniform(0, 1, (2, 1))])
            with quiet():
                r_up, r_us = up.step(a), ours.step(a)
            close_to(r_us[0], r_up[0], tol)
        np.testing.assert_allclose(ours._getAdjacencyMatrix(), up._getAdjacencyMatrix())
        up.close()


def test_controllers_agree_with_upstream_on_random_inputs():
    for model in ["cf2x", "cf2p", "racer"]:
        pairs = [(UpMRAC, compat.MRACControl), (UpCTBR, compat.CTBRControl)]
        if model != "racer":
            pairs.append((UpDSL, compat.DSLPIDControl))
        for up_cls, our_cls in pairs:
            up, ours = up_cls(up_enums.DroneModel(model)), our_cls(compat.DroneModel(model))
            rng = np.random.default_rng(11)
            # Same inputs, step by step, so internal state is compared too.
            for _ in range(40):
                state = np.zeros(20)
                state[0:3] = rng.normal(size=3)
                q = rng.normal(size=4)
                state[3:7] = q / np.linalg.norm(q)
                state[10:16] = rng.normal(scale=0.3, size=6)
                target, tvel = state[0:3] + rng.normal(scale=0.5, size=3), rng.normal(scale=0.2, size=3)
                with quiet():
                    a = up.computeControlFromState(1 / 48, state, target, target_vel=tvel)
                b = ours.computeControlFromState(1 / 48, state, target, target_vel=tvel)
                for x, y in zip(b, a):
                    np.testing.assert_allclose(x, y, rtol=1e-11, atol=1e-11)


NUDGE_KF = (
    "import gym_pybullet_drones.envs.BaseAviary as B\n"
    "_init = B.BaseAviary.__init__\n"
    "def _nudged(self, *a, **k):\n"
    "    _init(self, *a, **k)\n"
    "    self.KF = self.KF * (1 + 1e-12)\n"
    "B.BaseAviary.__init__ = _nudged\n"
)


def run_example(script, args, backend, out, folder):
    """Runs one of upstream's example scripts unmodified and returns its
    logged flight: on upstream, on upstream with KF changed by one part in
    1e12, or on dronelab.compat."""
    prelude = {"upstream": "", "nudged": NUDGE_KF, "compat": "import dronelab.compat as c\nc.install()\n"}[backend]
    args = args + ["--output_folder", folder]
    code = f"{prelude}import sys, runpy\nsys.argv = [{str(script)!r}] + {args!r}\nrunpy.run_path({str(script)!r}, run_name='__main__')\n"
    env = {**os.environ, "MPLBACKEND": "Agg"}
    res = subprocess.run([sys.executable, "-c", code], cwd=out, env=env, capture_output=True, text=True, timeout=600)
    assert res.returncode == 0, res.stderr[-3000:]
    logs = glob.glob(os.path.join(out, folder, "save-flight-*.npy"))
    if logs:
        return np.load(logs[0])["states"]
    # Some examples only write CSVs: one file per variable and drone.
    csvs = sorted(glob.glob(os.path.join(out, folder, "save-flight-*", "*.csv")))
    assert csvs, f"no flight log in {folder}"
    return np.stack([np.loadtxt(f, delimiter=",")[:, 1] for f in csvs])


@pytest.mark.parametrize("script,args", [
    ("pid.py", ["--gui", "False", "--plot", "False", "--obstacles", "False", "--duration_sec", "4"]),
    ("pid.py", ["--gui", "False", "--plot", "False", "--obstacles", "False", "--duration_sec", "4", "--physics", "dyn"]),
    ("pid_velocity.py", ["--gui", "False", "--plot", "False", "--obstacles", "False", "--duration_sec", "4"]),
    ("downwash.py", ["--gui", "False", "--duration_sec", "4"]),
])
def test_upstream_example_scripts_run_unmodified(tmp_path, script, args):
    """Upstream's own examples, unchanged, on both backends.

    These closed loops amplify rounding-level differences, so the check is
    that dronelab stays as close to upstream as upstream stays to itself
    when KF changes by one part in 1e12."""
    up = run_example(UPSTREAM_EXAMPLES / script, args, "upstream", tmp_path, "up")
    nudged = run_example(UPSTREAM_EXAMPLES / script, args, "nudged", tmp_path, "nudged")
    ours = run_example(UPSTREAM_EXAMPLES / script, args, "compat", tmp_path, "ours")
    assert ours.shape == up.shape
    scale = np.maximum(1.0, np.abs(up))
    t_axis = up.ndim - 1
    other = tuple(range(t_axis))
    ours_err = (np.abs(ours - up) / scale).max(axis=other)
    self_err = np.maximum.accumulate((np.abs(nudged - up) / scale).max(axis=other))
    # DYN is ported operation for operation. PYB stands in for Bullet's
    # integrator, which open loop differs from it ~1e4 times more than the
    # KF nudge does (mostly attitude), as in crates/compat/tests/envs.rs.
    floor, amplification = (1e-9, 1e2) if "dyn" in args else (1e-5, 1e4)
    print(f"{script} {' '.join(args)}: dronelab vs upstream {ours_err.max():.1e}, upstream vs itself {self_err.max():.1e}")
    bad = ours_err > np.maximum(floor, amplification * self_err)
    assert not bad.any(), f"first excess at log step {int(np.argmax(bad))}: {ours_err[bad][0]:.2e} vs self {self_err[bad][0]:.2e}"

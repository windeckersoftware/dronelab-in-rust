"""Record reference data from gym-pybullet-drones for dronelab-compat's tests.

Run with the pinned upstream installed (see crates/compat/README.md):

    bench/.venv-pybullet/bin/python crates/compat/tools/record_upstream.py

Writes crates/compat/tests/golden/*.json. Everything here is upstream's own
code; this script only chooses inputs and writes down what comes out.
"""

import io
import json
import contextlib
import importlib.metadata
from pathlib import Path

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation
from transforms3d.quaternions import mat2quat, qconjugate, qmult, rotate_vector

import gym_pybullet_drones
from gym_pybullet_drones.control.CTBRControl import CTBRControl
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.control.MRACControl import MRACControl
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
from gym_pybullet_drones.utils.enums import ActionType, DroneModel, ObservationType, Physics

OUT = Path(__file__).resolve().parents[1] / "tests" / "golden"
UPSTREAM_COMMIT = "01563120e060c9233a61f9cf00e6d4e5a8b331ab"
MODELS = [DroneModel.CF2X, DroneModel.CF2P, DroneModel.RACE]


def tolist(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if isinstance(x, (list, tuple)):
        return [tolist(v) for v in x]
    if isinstance(x, dict):
        return {k: tolist(v) for k, v in x.items()}
    return x


def save(name, data):
    meta = {
        "upstream": "learnsyslab/gym-pybullet-drones",
        "commit": UPSTREAM_COMMIT,
        "version": importlib.metadata.version("gym-pybullet-drones"),
        "pybullet": importlib.metadata.version("pybullet"),
        "numpy": np.__version__,
    }
    (OUT / f"{name}.json").write_text(json.dumps({"meta": meta, "data": tolist(data)}))
    print(f"wrote {name}.json ({(OUT / f'{name}.json').stat().st_size // 1024} KiB)")


@contextlib.contextmanager
def quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


class ForceRecorder:
    """Wraps PyBullet's force/torque calls to capture upstream's inputs."""

    def __init__(self):
        self.calls = []
        self._f, self._t = p.applyExternalForce, p.applyExternalTorque

    def __enter__(self):
        def force(body, link, forceObj, posObj, flags, physicsClientId):
            self.calls.append(["force", int(body), int(link), list(map(float, forceObj)), int(flags)])
            return self._f(body, link, forceObj, posObj, flags, physicsClientId=physicsClientId)

        def torque(body, link, torqueObj, flags, physicsClientId):
            self.calls.append(["torque", int(body), int(link), list(map(float, torqueObj)), int(flags)])
            return self._t(body, link, torqueObj, flags, physicsClientId=physicsClientId)

        p.applyExternalForce, p.applyExternalTorque = force, torque
        return self

    def __exit__(self, *exc):
        p.applyExternalForce, p.applyExternalTorque = self._f, self._t


def record_params():
    out = {}
    for m in MODELS:
        with quiet():
            env = CtrlAviary(drone_model=m, num_drones=3)
        out[m.value] = {k: getattr(env, k) for k in [
            "G", "M", "L", "THRUST2WEIGHT_RATIO", "J", "J_INV", "KF", "KM", "COLLISION_H", "COLLISION_R",
            "COLLISION_Z_OFFSET", "MAX_SPEED_KMH", "GND_EFF_COEFF", "PROP_RADIUS", "DRAG_COEFF", "DW_COEFF_1",
            "DW_COEFF_2", "DW_COEFF_3", "GRAVITY", "HOVER_RPM", "MAX_RPM", "MAX_THRUST", "MAX_XY_TORQUE",
            "MAX_Z_TORQUE", "GND_EFF_H_CLIP", "INIT_XYZS"]}
        env.close()
    save("params", out)


def record_math(rng):
    quats = rng.normal(size=(200, 4))
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    # A few near gimbal lock and exactly identity.
    special = [Rotation.from_euler("xyz", e).as_quat() for e in [[0, np.pi / 2, 0], [0.3, -np.pi / 2, 1.0], [0, 0, 0]]]
    quats = np.vstack([quats, special])
    eulers = rng.uniform(-np.pi, np.pi, size=(200, 3))
    eulers[:, 1] /= 2
    mats = [Rotation.from_quat(q).as_matrix() for q in quats]
    vec = rng.normal(size=(len(quats), 3))
    out = {
        "quat_xyzw": quats,
        "pb_matrix": [p.getMatrixFromQuaternion(q) for q in quats],
        "pb_euler": [p.getEulerFromQuaternion(q) for q in quats],
        "euler": eulers,
        "pb_quat_from_euler": [p.getQuaternionFromEuler(e) for e in eulers],
        "scipy_matrix_to_euler_XYZ": [Rotation.from_matrix(m).as_euler("XYZ") for m in mats],
        "scipy_euler_XYZ_to_quat": [Rotation.from_euler("XYZ", e).as_quat() for e in eulers],
        "scipy_euler_XYZ_to_matrix": [Rotation.from_euler("XYZ", e).as_matrix() for e in eulers],
        "scipy_quat_to_matrix": mats,
        "t3d_mat2quat_wxyz": [mat2quat(m) for m in mats],
        "vec": vec,
        "t3d_rotate_vector_wxyz": [rotate_vector(v, np.array([q[3], q[0], q[1], q[2]])) for v, q in zip(vec, quats)],
        "t3d_qmult_conj": [qmult(qconjugate(np.array([a[3], *a[:3]])), np.array([b[3], *b[:3]])) for a, b in zip(quats, quats[::-1])],
    }
    save("math", out)


def random_states(rng, n, scale=1.0):
    """A smooth random flight: positions, quaternions, velocities."""
    t = np.arange(n) / 48
    pos = np.stack([scale * np.sin(0.7 * t + rng.uniform(0, 6)) for _ in range(3)], 1) + [0, 0, 1]
    rpy = np.stack([0.4 * np.sin(1.3 * t + rng.uniform(0, 6)) for _ in range(3)], 1)
    rpy[:, 2] *= 4
    quat = np.array([p.getQuaternionFromEuler(e) for e in rpy])
    vel = np.gradient(pos, axis=0) * 48 + rng.normal(scale=0.05, size=pos.shape)
    angv = rng.normal(scale=0.3, size=pos.shape)
    return pos, quat, vel, angv


def record_controllers(rng):
    out = {"dsl": [], "mrac": [], "ctbr": []}
    for m in [DroneModel.CF2X, DroneModel.CF2P]:
        for case, scale in [("gentle", 0.3), ("aggressive", 3.0)]:
            ctrl = DSLPIDControl(drone_model=m)
            n = 150
            pos, quat, vel, angv = random_states(rng, n)
            tpos = pos + rng.normal(scale=scale, size=pos.shape)
            trpy = np.stack([np.zeros(n), np.zeros(n), rng.uniform(-3, 3, n)], 1)
            tvel = rng.normal(scale=0.5, size=pos.shape)
            trates = rng.normal(scale=0.2, size=pos.shape)
            steps = []
            for k in range(n):
                with quiet():
                    rpm, pos_e, yaw_e = ctrl.computeControl(1 / 48, pos[k], quat[k], vel[k], angv[k], tpos[k], trpy[k], tvel[k], trates[k])
                steps.append({"rpm": rpm, "pos_e": pos_e, "yaw_e": yaw_e})
            # setPIDCoefficients + reset path
            ctrl.setPIDCoefficients(p_coeff_pos=np.array([0.5, 0.5, 1.0]))
            ctrl.reset()
            with quiet():
                after = ctrl.computeControl(1 / 48, pos[0], quat[0], vel[0], angv[0], tpos[0])
            out["dsl"].append({
                "model": m.value, "case": case, "dt": 1 / 48, "pos": pos, "quat": quat, "vel": vel, "ang_vel": angv,
                "target_pos": tpos, "target_rpy": trpy, "target_vel": tvel, "target_rpy_rates": trates, "steps": steps,
                "after_set_pid_and_reset": {"rpm": after[0], "pos_e": after[1], "yaw_e": after[2]},
            })
    for m in MODELS:
        ctrl = MRACControl(drone_model=m)
        n = 120
        pos, quat, vel, angv = random_states(rng, n)
        tpos = pos + rng.normal(scale=0.3, size=pos.shape)
        steps = []
        for k in range(n):
            rpm, pos_e, rpy_e = ctrl.computeControl(1 / 48, pos[k], quat[k], vel[k], angv[k], tpos[k])
            steps.append({"rpm": rpm, "pos_e": pos_e, "rpy_e": rpy_e})
        fresh = MRACControl(drone_model=m)
        out["mrac"].append({
            "model": m.value, "dt": 1 / 48, "pos": pos, "quat": quat, "vel": vel, "ang_vel": angv, "target_pos": tpos,
            "steps": steps, "Kx": fresh.Kx, "Kr": fresh.Kr, "Kr_ref_gain": fresh.Kr_ref_gain, "Am": fresh.Am,
            "Bm": fresh.Bm, "P": fresh.P, "Gamma_x": fresh.Gamma_x, "Gamma_r": fresh.Gamma_r,
        })
    for m in MODELS:
        ctrl = CTBRControl(drone_model=m)
        pos, quat, vel, angv = random_states(rng, 100)
        quat_wxyz = np.array([[q[3], q[0], q[1], q[2]] for q in quat])
        tpos = pos + rng.normal(scale=0.5, size=pos.shape)
        tvel = rng.normal(scale=0.5, size=pos.shape)
        outs = [ctrl.computeControl(1 / 48, pos[k], quat_wxyz[k], vel[k], angv[k], tpos[k], np.zeros(3), tvel[k]) for k in range(100)]
        out["ctbr"].append({"model": m.value, "pos": pos, "quat_wxyz": quat_wxyz, "vel": vel,
                                                   "ang_vel": angv, "target_pos": tpos, "target_vel": tvel, "out": outs})
    save("controllers", out)


def rpm_sequence(rng, env, n, spread):
    base = env.HOVER_RPM
    walk = np.cumsum(rng.normal(scale=spread / 10, size=(n, env.NUM_DRONES, 4)), 0)
    return np.clip(base * (1 + np.clip(walk, -spread, spread) + rng.normal(scale=spread / 5, size=walk.shape)), 0, env.MAX_RPM)


def run_ctrl_aviary(model, physics, n_drones, n_steps, rng, pyb_freq=240, ctrl_freq=48, spread=0.05, initial=None):
    with quiet():
        env = CtrlAviary(drone_model=model, num_drones=n_drones, physics=physics, pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq, initial_xyzs=initial)
        obs, _ = env.reset(seed=0)
    acts = rpm_sequence(rng, env, n_steps, spread)
    obs_seq, forces = [obs], []
    for a in acts:
        with ForceRecorder() as rec, quiet():
            obs, *_ = env.step(a)
        obs_seq.append(obs)
        forces.append(rec.calls)
    env.close()
    return {"model": model.value, "physics": physics.value, "num_drones": n_drones, "pyb_freq": pyb_freq,
            "ctrl_freq": ctrl_freq, "initial_xyzs": initial, "actions": acts, "obs": obs_seq, "forces": forces}


def record_physics(rng):
    out = []
    # Explicit dynamics: exact comparison.
    for m in MODELS:
        out.append(run_ctrl_aviary(m, Physics.DYN, 2, 60, rng, spread=0.08))
    # PyBullet engine modes, airborne (start at 1 m) and from the ground.
    lift = lambda n: np.array([[i * 0.5, 0, 1.0] for i in range(n)])
    stack = np.array([[0, 0, 0.6], [0.05, 0, 1.0], [-0.05, 0.05, 1.4]])  # for downwash
    for m in MODELS:
        for phys in [Physics.PYB, Physics.PYB_DRAG]:
            out.append(run_ctrl_aviary(m, phys, 1, 48, rng, initial=lift(1)))
    out.append(run_ctrl_aviary(DroneModel.CF2X, Physics.PYB, 1, 96, rng))                      # take-off from ground
    out.append(run_ctrl_aviary(DroneModel.CF2X, Physics.PYB_GND, 1, 96, rng))                  # ground effect
    out.append(run_ctrl_aviary(DroneModel.CF2X, Physics.PYB_DW, 3, 48, rng, initial=stack))     # downwash
    out.append(run_ctrl_aviary(DroneModel.CF2X, Physics.PYB_GND_DRAG_DW, 3, 48, rng, initial=stack))
    save("physics", out)


def run_rl(env_cls, kwargs, actions_fn, n_steps):
    with quiet():
        env = env_cls(**kwargs)
        obs, info = env.reset(seed=0)
    steps = []
    for k in range(n_steps):
        a = actions_fn(k, env)
        with quiet():
            obs2, r, term, trunc, info = env.step(a)
        steps.append({"action": a, "obs": obs2, "reward": r, "terminated": term, "truncated": trunc})
        if term or trunc:
            break
    env.close()
    # Upstream against itself: the same episode with the rotor thrust
    # coefficient changed by one part in 1e12, a small error injected at
    # every physics step like rounding differences are. Where this grows,
    # the episode is chaotic at rounding level and no reimplementation can
    # match it more closely.
    with quiet():
        env = env_cls(**kwargs)
        env.KF = env.KF * (1 + 1e-12)
        env.reset(seed=0)
    divergence = []
    for st in steps:
        with quiet():
            o, *_ = env.step(np.array(st["action"], dtype=float))
        divergence.append(float(np.abs(np.asarray(o, float) - np.asarray(st["obs"], float)).max()))
    env.close()
    return {"initial_obs": obs, "steps": steps, "self_divergence": divergence}


def record_envs(rng):
    out = []
    for physics in [Physics.DYN, Physics.PYB]:
        for act in [ActionType.RPM, ActionType.PID, ActionType.VEL, ActionType.ONE_D_RPM, ActionType.ONE_D_PID]:
            for cls, extra in [(HoverAviary, {}), (MultiHoverAviary, {"num_drones": 2})]:
                with quiet():
                    probe = cls(physics=physics, act=act, **extra)
                shape = probe.action_space.shape
                probe.close()
                local = np.random.default_rng(len(out))
                if act == ActionType.PID:
                    fn = lambda k, env, local=local, shape=shape: np.clip(np.array([[0, 0, 1.0]] * shape[0]) + local.normal(scale=0.2, size=shape), -1, 1)
                else:
                    fn = lambda k, env, local=local, shape=shape: local.uniform(-1, 1, size=shape)
                n = 260 if (physics == Physics.DYN and act == ActionType.ONE_D_RPM and cls is HoverAviary) else 60
                rec = run_rl(cls, dict(physics=physics, act=act, **extra), fn, n)
                rec.update({"env": cls.__name__, "physics": physics.value, "act": act.value, "kwargs": extra})
                out.append(rec)
    # VelocityAviary (Velocity commands through DSLPIDControl).
    for physics in [Physics.DYN, Physics.PYB]:
        local = np.random.default_rng(99)
        fn = lambda k, env, local=local: np.hstack([local.uniform(-1, 1, (2, 3)), local.uniform(0, 1, (2, 1))])
        rec = run_rl(VelocityAviary, dict(physics=physics, num_drones=2, ctrl_freq=48), fn, 60)
        rec.update({"env": "VelocityAviary", "physics": physics.value, "act": "vel", "kwargs": {"num_drones": 2, "ctrl_freq": 48}})
        out.append(rec)
    save("envs", out)


def main():
    import sys

    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261016)
    only = set(sys.argv[1:])
    for name, fn in [("params", lambda: record_params()), ("math", lambda: record_math(rng)),
                     ("controllers", lambda: record_controllers(rng)), ("physics", lambda: record_physics(rng)),
                     ("envs", lambda: record_envs(rng))]:
        if not only or name in only:
            fn()


if __name__ == "__main__":
    main()

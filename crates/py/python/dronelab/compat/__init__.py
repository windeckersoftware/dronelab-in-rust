"""Drop-in replacement for gym-pybullet-drones (v2.2.0), stepped in Rust.

The module layout mirrors upstream, so changing

    from gym_pybullet_drones.envs.HoverAviary import HoverAviary

to

    from dronelab.compat.envs.HoverAviary import HoverAviary

is enough. Or call ``dronelab.compat.install()`` once and unmodified
upstream scripts import this package instead.

Every formula follows upstream, checked against outputs recorded from
upstream itself (see crates/compat). ``Physics.DYN`` matches to rounding
error. The ``PYB*`` modes apply exactly upstream's forces and stand in
for PyBullet's integrator (about 1e-9 m over a second of free flight);
ground contact is only approximate. There is no GUI, recording or camera
observations.
"""

import sys

from .control import BaseControl, CTBRControl, DSLPIDControl, MRACControl
from .envs import BaseAviary, BaseRLAviary, CtrlAviary, HoverAviary, MultiHoverAviary, VelocityAviary
from .utils.enums import ActionType, DroneModel, ImageType, ObservationType, Physics

__all__ = [
    "ActionType", "BaseAviary", "BaseControl", "BaseRLAviary", "CTBRControl", "CtrlAviary", "DSLPIDControl",
    "DroneModel", "HoverAviary", "ImageType", "MRACControl", "MultiHoverAviary", "ObservationType", "Physics",
    "VelocityAviary", "install",
]

UPSTREAM_VERSION = "2.2.0"
UPSTREAM_COMMIT = "01563120e060c9233a61f9cf00e6d4e5a8b331ab"


def _register():
    from gymnasium.envs.registration import register, registry

    for env_id, cls in [("ctrl-aviary-v0", "CtrlAviary"), ("velocity-aviary-v0", "VelocityAviary"),
                        ("hover-aviary-v0", "HoverAviary"), ("multihover-aviary-v0", "MultiHoverAviary")]:
        full = f"dronelab/{env_id}"
        if full not in registry:
            register(id=full, entry_point=f"dronelab.compat.envs:{cls}")


_register()


def install():
    """Make ``import gym_pybullet_drones...`` load this package.

    For running existing scripts unchanged. Refuses if the real package is
    already imported.
    """
    import importlib

    current = sys.modules.get("gym_pybullet_drones")
    if current is not None and getattr(current, "__dronelab_compat__", False) is False:
        raise RuntimeError("gym_pybullet_drones is already imported; call install() before importing it")
    names = [
        "", ".envs", ".envs.BaseAviary", ".envs.BaseRLAviary", ".envs.CtrlAviary", ".envs.VelocityAviary",
        ".envs.HoverAviary", ".envs.MultiHoverAviary", ".control", ".control.BaseControl",
        ".control.DSLPIDControl", ".control.CTBRControl", ".control.MRACControl", ".utils", ".utils.enums",
        ".utils.utils", ".utils.Logger",
    ]
    for suffix in names:
        try:
            module = importlib.import_module(f"dronelab.compat{suffix}")
        except ImportError:  # e.g. Logger without matplotlib
            continue
        sys.modules[f"gym_pybullet_drones{suffix}"] = module
    sys.modules["gym_pybullet_drones"].__dronelab_compat__ = True


__dronelab_compat__ = True

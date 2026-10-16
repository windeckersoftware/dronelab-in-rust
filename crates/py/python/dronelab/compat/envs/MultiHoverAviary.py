import numpy as np

from .BaseRLAviary import BaseRLAviary
from ..utils.enums import ActionType, DroneModel, ObservationType, Physics


class MultiHoverAviary(BaseRLAviary):
    """Multi-agent RL problem: leader-follower."""

    _ENV = "multihover"

    def __init__(self, drone_model: DroneModel = DroneModel.CF2X, num_drones: int = 2, neighbourhood_radius: float = np.inf,
                 initial_xyzs=None, initial_rpys=None, physics: Physics = Physics.PYB, pyb_freq: int = 240,
                 ctrl_freq: int = 30, gui=False, record=False, obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.RPM):
        super().__init__(drone_model=drone_model, num_drones=num_drones, neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs, initial_rpys=initial_rpys, physics=physics, pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq, gui=gui, record=record, obs=obs, act=act)

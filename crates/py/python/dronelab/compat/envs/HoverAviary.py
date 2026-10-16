import numpy as np

from .BaseRLAviary import BaseRLAviary
from ..utils.enums import ActionType, DroneModel, ObservationType, Physics


class HoverAviary(BaseRLAviary):
    """Single agent RL problem: hover at position [0, 0, 1]."""

    _ENV = "hover"

    def __init__(self, drone_model: DroneModel = DroneModel.CF2X, initial_xyzs=None, initial_rpys=None,
                 physics: Physics = Physics.PYB, pyb_freq: int = 240, ctrl_freq: int = 30, gui=False, record=False,
                 obs: ObservationType = ObservationType.KIN, act: ActionType = ActionType.RPM):
        super().__init__(drone_model=drone_model, num_drones=1, initial_xyzs=initial_xyzs, initial_rpys=initial_rpys,
                         physics=physics, pyb_freq=pyb_freq, ctrl_freq=ctrl_freq, gui=gui, record=record, obs=obs, act=act)

    def _computeReward(self):
        state = self._getDroneStateVector(0)
        return max(0, 2 - np.linalg.norm(self.TARGET_POS - state[0:3]) ** 4)

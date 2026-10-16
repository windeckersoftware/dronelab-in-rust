import numpy as np
from gymnasium import spaces

from .BaseAviary import BaseAviary
from ..utils.enums import ActionType, DroneModel, ObservationType, Physics, value_of


class BaseRLAviary(BaseAviary):
    """Base single and multi-agent environment class for reinforcement learning.

    Only ``ObservationType.KIN`` is available. As upstream, ``reset()`` keeps
    the PID controllers' state and the action buffer.
    """

    def __init__(self, drone_model: DroneModel = DroneModel.CF2X, num_drones: int = 1, neighbourhood_radius: float = np.inf,
                 initial_xyzs=None, initial_rpys=None, physics: Physics = Physics.PYB, pyb_freq: int = 240,
                 ctrl_freq: int = 240, gui=False, record=False, obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.RPM):
        self.ACTION_BUFFER_SIZE = int(ctrl_freq // 2)
        self.OBS_TYPE = ObservationType(value_of(obs))
        self.ACT_TYPE = ActionType(value_of(act))
        if self.OBS_TYPE != ObservationType.KIN:
            raise NotImplementedError("dronelab.compat only has ObservationType.KIN (no cameras)")
        super().__init__(drone_model=drone_model, num_drones=num_drones, neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs, initial_rpys=initial_rpys, physics=physics, pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq, gui=gui, record=record, obstacles=True, user_debug_gui=False,
                         vision_attributes=False, _act=self.ACT_TYPE.value)

    def _act_size(self):
        return {ActionType.RPM: 4, ActionType.VEL: 4, ActionType.PID: 3, ActionType.ONE_D_RPM: 1, ActionType.ONE_D_PID: 1}[self.ACT_TYPE]

    def _actionSpace(self):
        size = self._act_size()
        lo = np.array([-1 * np.ones(size) for i in range(self.NUM_DRONES)])
        hi = np.array([+1 * np.ones(size) for i in range(self.NUM_DRONES)])
        return spaces.Box(low=lo, high=hi, dtype=np.float32)

    def _observationSpace(self):
        lo, hi = -np.inf, np.inf
        obs_lo = np.array([[lo, lo, 0, lo, lo, lo, lo, lo, lo, lo, lo, lo] for i in range(self.NUM_DRONES)])
        obs_hi = np.array([[hi, hi, hi, hi, hi, hi, hi, hi, hi, hi, hi, hi] for i in range(self.NUM_DRONES)])
        size = self._act_size()
        for i in range(self.ACTION_BUFFER_SIZE):
            obs_lo = np.hstack([obs_lo, -np.ones((self.NUM_DRONES, size))])
            obs_hi = np.hstack([obs_hi, np.ones((self.NUM_DRONES, size))])
        return spaces.Box(low=obs_lo, high=obs_hi, dtype=np.float32)

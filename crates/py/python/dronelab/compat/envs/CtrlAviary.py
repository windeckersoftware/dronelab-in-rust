import numpy as np
from gymnasium import spaces

from .BaseAviary import BaseAviary
from ..utils.enums import DroneModel, Physics


class CtrlAviary(BaseAviary):
    """Multi-drone environment class for control applications (RPM in)."""

    _ENV = "ctrl"

    def __init__(self, drone_model: DroneModel = DroneModel.CF2X, num_drones: int = 1, neighbourhood_radius: float = np.inf,
                 initial_xyzs=None, initial_rpys=None, physics: Physics = Physics.PYB, pyb_freq: int = 240,
                 ctrl_freq: int = 240, gui=False, record=False, obstacles=False, user_debug_gui=True, output_folder="results"):
        super().__init__(drone_model=drone_model, num_drones=num_drones, neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs, initial_rpys=initial_rpys, physics=physics, pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq, gui=gui, record=record, obstacles=obstacles, user_debug_gui=user_debug_gui,
                         output_folder=output_folder)

    def _actionSpace(self):
        lo = np.array([[0., 0., 0., 0.] for i in range(self.NUM_DRONES)])
        hi = np.array([[self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM] for i in range(self.NUM_DRONES)])
        return spaces.Box(low=lo, high=hi, dtype=np.float32)

    def _observationSpace(self):
        lo = np.array([[-np.inf, -np.inf, 0., -1., -1., -1., -1., -np.pi, -np.pi, -np.pi, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0., 0., 0., 0.] for i in range(self.NUM_DRONES)])
        hi = np.array([[np.inf, np.inf, np.inf, 1., 1., 1., 1., np.pi, np.pi, np.pi, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM] for i in range(self.NUM_DRONES)])
        return spaces.Box(low=lo, high=hi, dtype=np.float32)

    def _computeReward(self):
        return -1

    def _computeTerminated(self):
        return False

    def _computeTruncated(self):
        return False

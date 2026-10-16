import numpy as np

from ... import _native
from .BaseControl import BaseControl, _vec
from ..utils.enums import DroneModel, value_of


class MRACControl(BaseControl):
    """Model-reference adaptive control (gym_pybullet_drones' MRACControl).

    The reference model's matrices were computed by upstream's own code
    (pole placement and a Lyapunov solve) and are built in.
    """

    def __init__(self, drone_model: DroneModel, g: float = 9.8):
        model = DroneModel(value_of(drone_model))
        self._sim = _native.Mrac(model.value, g)
        super().__init__(drone_model=model, g=g)
        self.g = g

    def reset(self):
        self._sim.reset()

    @property
    def control_counter(self):
        return self._sim.control_counter

    @control_counter.setter
    def control_counter(self, value):
        if value != 0:
            raise AttributeError("control_counter can only be reset")
        self._sim.reset()

    def computeControl(self, control_timestep, cur_pos, cur_quat, cur_vel, cur_ang_vel, target_pos,
                       target_rpy=np.zeros(3), target_vel=np.zeros(3), target_rpy_rates=np.zeros(3)):
        """Returns ``(rpm, pos_e, rpy_e)``, as upstream."""
        return self._sim.compute(float(control_timestep), _vec(cur_pos), _vec(cur_quat, 4), _vec(cur_vel),
                                 _vec(cur_ang_vel), _vec(target_pos), _vec(target_rpy), _vec(target_vel),
                                 _vec(target_rpy_rates))

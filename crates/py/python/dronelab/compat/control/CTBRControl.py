import numpy as np

from ... import _native
from .BaseControl import _vec
from ..utils.enums import DroneModel, value_of


class CTBRControl(object):
    """Collective thrust and body rates (gym_pybullet_drones' CTBRControl).

    As upstream, ``cur_quat`` is ``[w, x, y, z]``.
    """

    def __init__(self, drone_model: DroneModel, g: float = 9.8):
        self.DRONE_MODEL = DroneModel(value_of(drone_model))
        self._sim = _native.Ctbr(self.DRONE_MODEL.value, g)
        self.GRAVITY, self.KF, self.KM = self._sim.constants
        self.reset()

    def reset(self):
        self.control_counter = 0
        self._sim.reset()

    def computeControlFromState(self, control_timestep, state, target_pos, target_rpy=np.zeros(3),
                                target_vel=np.zeros(3), target_rpy_rates=np.zeros(3)):
        return self.computeControl(control_timestep=control_timestep, cur_pos=state[0:3],
                                   cur_quat=np.array([state[6], state[3], state[4], state[5]]), cur_vel=state[10:13],
                                   cur_ang_vel=state[13:16], target_pos=target_pos, target_rpy=target_rpy,
                                   target_vel=target_vel, target_rpy_rates=target_rpy_rates)

    def computeControl(self, control_timestep, cur_pos, cur_quat, cur_vel, cur_ang_vel, target_pos,
                       target_rpy=np.zeros(3), target_vel=np.zeros(3), target_rpy_rates=np.zeros(3)):
        """Returns ``(normalised thrust, roll rate, pitch rate, yaw rate)``."""
        return self._sim.compute(_vec(cur_pos), _vec(cur_quat, 4), _vec(cur_vel), _vec(target_pos), _vec(target_vel))

    def setPIDCoefficients(self, *args, **kwargs):
        raise AttributeError("in BaseControl.setPIDCoefficients(), not all PID coefficients exist as attributes in the instantiated control class.")

import numpy as np

from ... import _native
from ..utils.enums import DroneModel, value_of


class BaseControl(object):
    """Base class for control (gym_pybullet_drones.control.BaseControl)."""

    def __init__(self, drone_model: DroneModel, g: float = 9.8):
        self.DRONE_MODEL = DroneModel(value_of(drone_model))
        self.GRAVITY = g * self._getURDFParameter("m")
        self.KF = self._getURDFParameter("kf")
        self.KM = self._getURDFParameter("km")
        self.reset()

    def reset(self):
        self.control_counter = 0

    def computeControlFromState(self, control_timestep, state, target_pos, target_rpy=np.zeros(3),
                                target_vel=np.zeros(3), target_rpy_rates=np.zeros(3)):
        return self.computeControl(control_timestep=control_timestep, cur_pos=state[0:3], cur_quat=state[3:7],
                                   cur_vel=state[10:13], cur_ang_vel=state[13:16], target_pos=target_pos,
                                   target_rpy=target_rpy, target_vel=target_vel, target_rpy_rates=target_rpy_rates)

    def computeControl(self, control_timestep, cur_pos, cur_quat, cur_vel, cur_ang_vel, target_pos,
                       target_rpy=np.zeros(3), target_vel=np.zeros(3), target_rpy_rates=np.zeros(3)):
        raise NotImplementedError

    def setPIDCoefficients(self, p_coeff_pos=None, i_coeff_pos=None, d_coeff_pos=None, p_coeff_att=None,
                           i_coeff_att=None, d_coeff_att=None):
        ATTR_LIST = ["P_COEFF_FOR", "I_COEFF_FOR", "D_COEFF_FOR", "P_COEFF_TOR", "I_COEFF_TOR", "D_COEFF_TOR"]
        if not all(hasattr(self, attr) for attr in ATTR_LIST):
            raise AttributeError("in BaseControl.setPIDCoefficients(), not all PID coefficients exist as attributes in the instantiated control class.")
        self.P_COEFF_FOR = self.P_COEFF_FOR if p_coeff_pos is None else p_coeff_pos
        self.I_COEFF_FOR = self.I_COEFF_FOR if i_coeff_pos is None else i_coeff_pos
        self.D_COEFF_FOR = self.D_COEFF_FOR if d_coeff_pos is None else d_coeff_pos
        self.P_COEFF_TOR = self.P_COEFF_TOR if p_coeff_att is None else p_coeff_att
        self.I_COEFF_TOR = self.I_COEFF_TOR if i_coeff_att is None else i_coeff_att
        self.D_COEFF_TOR = self.D_COEFF_TOR if d_coeff_att is None else d_coeff_att

    def _getURDFParameter(self, parameter_name: str):
        return _native.urdf_parameter(self.DRONE_MODEL.value, parameter_name)


def _vec(x, n=3):
    return np.ascontiguousarray(np.asarray(x, dtype=np.float64).reshape(n))

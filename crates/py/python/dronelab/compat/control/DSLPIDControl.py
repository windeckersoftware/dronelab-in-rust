import numpy as np

from ... import _native
from .BaseControl import BaseControl, _vec
from ..utils.enums import DroneModel, value_of


def _gain(name, arg):
    def get(self):
        return self._sim.coefficients()[name]

    def set(self, value):
        self._sim.set_pid(**{arg: _vec(value)})

    return property(get, set)


class DSLPIDControl(BaseControl):
    """PID control class for Crazyflies (gym_pybullet_drones' DSLPIDControl)."""

    P_COEFF_FOR = _gain("P_COEFF_FOR", "p_coeff_pos")
    I_COEFF_FOR = _gain("I_COEFF_FOR", "i_coeff_pos")
    D_COEFF_FOR = _gain("D_COEFF_FOR", "d_coeff_pos")
    P_COEFF_TOR = _gain("P_COEFF_TOR", "p_coeff_att")
    I_COEFF_TOR = _gain("I_COEFF_TOR", "i_coeff_att")
    D_COEFF_TOR = _gain("D_COEFF_TOR", "d_coeff_att")

    def __init__(self, drone_model: DroneModel, g: float = 9.8):
        model = DroneModel(value_of(drone_model))
        if model not in (DroneModel.CF2X, DroneModel.CF2P):
            raise ValueError("in DSLPIDControl.__init__(), DSLPIDControl requires DroneModel.CF2X or DroneModel.CF2P")
        self._sim = _native.DslPid(model.value, g)
        super().__init__(drone_model=model, g=g)
        self.PWM2RPM_SCALE = 0.2685
        self.PWM2RPM_CONST = 4070.3
        self.MIN_PWM = 20000
        self.MAX_PWM = 65535
        if model == DroneModel.CF2X:
            self.MIXER_MATRIX = np.array([[-.5, -.5, -1], [-.5, .5, 1], [.5, .5, -1], [.5, -.5, 1]])
        else:
            self.MIXER_MATRIX = np.array([[0, -1, -1], [+1, 0, 1], [0, 1, -1], [-1, 0, 1]])

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
        """Returns ``(rpm, pos_e, yaw_e)``, as upstream."""
        return self._sim.compute(float(control_timestep), _vec(cur_pos), _vec(cur_quat, 4), _vec(cur_vel),
                                 _vec(cur_ang_vel), _vec(target_pos), _vec(target_rpy), _vec(target_vel),
                                 _vec(target_rpy_rates))

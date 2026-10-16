"""BaseAviary: the shared core of every aviary, stepped in Rust."""

import time
import warnings

import gymnasium as gym
import numpy as np

from ... import _native
from ..utils.enums import DroneModel, Physics, value_of


def _rows(a, n, what):
    if a is None:
        return None
    a = np.asarray(a, dtype=np.float64)
    if a.shape != (n, 3):
        raise ValueError(f"invalid {what} in BaseAviary.__init__(), try {what}.reshape(NUM_DRONES,3)")
    return np.ascontiguousarray(a)


class BaseAviary(gym.Env):
    """Base class for "drone aviary" Gym environments.

    Same constructor and attributes as gym_pybullet_drones' BaseAviary. The
    simulation runs in Rust; there is no PyBullet client, GUI, video
    recording or camera.
    """

    metadata = {"render.modes": ["human"]}
    _ENV = None  # which upstream class: "ctrl", "velocity", "hover", "multihover"

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 num_drones: int = 1,
                 neighbourhood_radius: float = np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 240,
                 gui=False,
                 record=False,
                 obstacles=False,
                 user_debug_gui=True,
                 vision_attributes=False,
                 output_folder="results",
                 _act="rpm",
                 ):
        if gui or record:
            raise NotImplementedError("dronelab.compat has no PyBullet GUI or video recording; use gym_pybullet_drones for those")
        if vision_attributes:
            raise NotImplementedError("dronelab.compat has no camera observations")
        if self._ENV is None:
            raise TypeError("BaseAviary is abstract; use CtrlAviary, VelocityAviary, HoverAviary or MultiHoverAviary")
        if obstacles and self._ENV in ("ctrl", "velocity"):
            warnings.warn("dronelab.compat does not simulate PyBullet's sample obstacles; drones cannot hit them", stacklevel=2)
        self.G = 9.8
        self.RAD2DEG = 180 / np.pi
        self.DEG2RAD = np.pi / 180
        self.CTRL_FREQ = ctrl_freq
        self.PYB_FREQ = pyb_freq
        self.NUM_DRONES = num_drones
        self.NEIGHBOURHOOD_RADIUS = neighbourhood_radius
        self.DRONE_MODEL = DroneModel(value_of(drone_model))
        self.GUI = gui
        self.RECORD = record
        self.PHYSICS = Physics(value_of(physics))
        self.OBSTACLES = obstacles
        self.USER_DEBUG = user_debug_gui
        self.URDF = self.DRONE_MODEL.value + ".urdf"
        self.OUTPUT_FOLDER = output_folder
        self.VISION_ATTR = vision_attributes
        self._sim = _native.CompatAviary(
            self._ENV, _act, self.DRONE_MODEL.value, num_drones, float(neighbourhood_radius),
            _rows(initial_xyzs, num_drones, "initial_xyzs"), _rows(initial_rpys, num_drones, "initial_rpys"),
            self.PHYSICS.value, pyb_freq, ctrl_freq,
        )
        for k, v in self._sim.constants().items():
            setattr(self, k, v)
        self.DRAG_COEFF = np.asarray(self.DRAG_COEFF)
        if initial_xyzs is not None:
            self.INIT_XYZS = initial_xyzs
        if initial_rpys is not None:
            self.INIT_RPYS = initial_rpys
        self.action_space = self._actionSpace()
        self.observation_space = self._observationSpace()
        self.first_render_call = True
        self.RESET_TIME = time.time()
        self._sync()

    # ------------------------------------------------------------------ state
    def _sync(self):
        k = self._sim.kinematics()
        self.pos, self.quat, self.rpy, self.vel, self.ang_v = k["pos"], k["quat"], k["rpy"], k["vel"], k["ang_v"]
        self.last_clipped_action = k["last_clipped_action"]
        self.step_counter = self._sim.step_counter

    def _getDroneStateVector(self, nth_drone):
        return self._sim.state_vector(nth_drone)

    def _getAdjacencyMatrix(self):
        return self._sim.adjacency_matrix()

    # ------------------------------------------------------------------- gym
    def reset(self, seed: int = None, options: dict = None):
        super().reset(seed=seed, options=options)
        obs = self._sim.reset()
        self.RESET_TIME = time.time()
        self.first_render_call = True
        self._sync()
        return obs, self._computeInfo()

    def step(self, action):
        a = np.asarray(action, dtype=np.float64).reshape(self.NUM_DRONES, -1)
        obs, reward, terminated, truncated = self._sim.step(np.ascontiguousarray(a))
        self._sync()
        return obs, reward, terminated, truncated, self._computeInfo()

    def render(self, mode="human", close=False):
        if self.first_render_call and not self.GUI:
            print("[WARNING] BaseAviary.render() is implemented as text-only, re-initialize the environment using Aviary(gui=True) to use PyBullet's graphical interface")
            self.first_render_call = False
        print("\n[INFO] BaseAviary.render() ——— it {:04d}".format(self.step_counter),
              "——— wall-clock time {:.1f}s,".format(time.time() - self.RESET_TIME),
              "simulation time {:.1f}s@{:d}Hz ({:.2f}x)".format(self.step_counter * self.PYB_TIMESTEP, self.PYB_FREQ, (self.step_counter * self.PYB_TIMESTEP) / max(time.time() - self.RESET_TIME, 1e-9)))
        for i in range(self.NUM_DRONES):
            print("[INFO] BaseAviary.render() ——— drone {:d}".format(i),
                  "——— x {:+06.2f}, y {:+06.2f}, z {:+06.2f}".format(self.pos[i, 0], self.pos[i, 1], self.pos[i, 2]),
                  "——— velocity {:+06.2f}, {:+06.2f}, {:+06.2f}".format(self.vel[i, 0], self.vel[i, 1], self.vel[i, 2]),
                  "——— roll {:+06.2f}, pitch {:+06.2f}, yaw {:+06.2f}".format(self.rpy[i, 0] * self.RAD2DEG, self.rpy[i, 1] * self.RAD2DEG, self.rpy[i, 2] * self.RAD2DEG),
                  "——— angular velocity {:+06.4f}, {:+06.4f}, {:+06.4f} ——— ".format(self.ang_v[i, 0], self.ang_v[i, 1], self.ang_v[i, 2]))

    def close(self):
        pass

    def getPyBulletClient(self):
        """There is no PyBullet client; returns -1 so scripts that only store it run."""
        return -1

    def getDroneIds(self):
        return np.arange(1, self.NUM_DRONES + 1)

    def _calculateNextStep(self, current_position, destination, step_size=1):
        direction = destination - current_position
        distance = np.linalg.norm(direction)
        if distance <= step_size:
            return destination
        return current_position + direction / distance * step_size

    def _computeInfo(self):
        return {"answer": 42}

    def _actionSpace(self):
        raise NotImplementedError

    def _observationSpace(self):
        raise NotImplementedError

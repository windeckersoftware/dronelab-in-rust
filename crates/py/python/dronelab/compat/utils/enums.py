"""The enums of gym_pybullet_drones.utils.enums, with the same values."""

from enum import Enum


class DroneModel(Enum):
    """Drone models enumeration class."""

    CF2X = "cf2x"   # Bitcraze Craziflie 2.0 in the X configuration
    CF2P = "cf2p"   # Bitcraze Craziflie 2.0 in the + configuration
    RACE = "racer"  # Racer drone in the X configuration


class Physics(Enum):
    """Physics implementations enumeration class."""

    PYB = "pyb"
    DYN = "dyn"
    PYB_GND = "pyb_gnd"
    PYB_DRAG = "pyb_drag"
    PYB_DW = "pyb_dw"
    PYB_GND_DRAG_DW = "pyb_gnd_drag_dw"


class ImageType(Enum):
    """Camera capture image type enumeration class."""

    RGB = 0
    DEP = 1
    SEG = 2
    BW = 3


class ActionType(Enum):
    """Action type enumeration class."""

    RPM = "rpm"
    PID = "pid"
    VEL = "vel"
    ONE_D_RPM = "one_d_rpm"
    ONE_D_PID = "one_d_pid"


class ObservationType(Enum):
    """Observation type enumeration class."""

    KIN = "kin"
    RGB = "rgb"
    DEP = "dep"
    ALL = "all"


def value_of(e):
    """The string value of one of these enums, or of upstream's own."""
    return e.value if isinstance(e, Enum) else str(e)

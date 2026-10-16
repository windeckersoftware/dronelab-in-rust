"""gym_pybullet_drones.utils.utils: wall-clock sync and argument parsing."""

import argparse
import time


def sync(i, start_time, timestep):
    """Sleeps so that iteration ``i`` happens no earlier than ``i * timestep``."""
    if timestep > .04 or i % (int(1 / (24 * timestep))) == 0:
        elapsed = time.time() - start_time
        if elapsed < (i * timestep):
            time.sleep(timestep * i - elapsed)


def str2bool(val):
    """Converts a string into a boolean."""
    if isinstance(val, bool):
        return val
    elif val.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif val.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("[ERROR] in str2bool(), a Boolean value is expected")

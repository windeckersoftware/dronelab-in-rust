# dronelab

A fast, deterministic multirotor and eVTOL simulator for reinforcement
learning, written in Rust, after
[gym-pybullet-drones](https://github.com/learnsyslab/gym-pybullet-drones).
It comes with a Gymnasium / Stable-Baselines3 Python API, and its flight
controller is the same `no_std` code that runs on an STM32.

```
crates/core   dronelab-core  no_std flight code: math, mixer, PID, BaseControl port
crates/sim    dronelab       simulator: dynamics, motors, battery, IMU, wind,
                             randomization, tasks, racing, rayon batch
crates/py     dronelab-py    pyo3 module + Python package `dronelab`
```

## Airframes

| name | rotors | mass | notes |
|---|---|---|---|
| `crazyflie_x` | 4 | 27 g | from gym-pybullet-drones' `cf2x.urdf` |
| `racer` | 4 | 0.83 kg | from `racer.urdf` |
| `evtol_single_seat` | 8 (X8 coaxial) | 200 kg | ~26 kW hover, 96S 34 Ah pack |
| `evtol_two_seat` | 16 (8 coaxial arms) | 620 kg | ~80 kW hover, 104S 62 Ah pack |

The eVTOL numbers are illustrative, not a specific aircraft. Add your own
vehicle by building a `VehicleParams` (rotor positions, spins, max thrust)
and an `Airframe` (rotor speed, prop size and pitch, motor time constants,
battery, drag).

## What it models

`Realism::FULL` (default) turns on everything; `Realism::IDEAL` is close to
gym-pybullet-drones' explicit `DYN` physics.

- **Motor lag**: first-order spin-up and spin-down. Big eVTOL props take
  0.15–0.25 s, and that is what limits how fast the vehicle can be controlled.
- **Battery**: open-circuit voltage curve and internal resistance. Under load
  the voltage sags, and the top rotor speed drops with it.
- **Prop thrust curve**: blade-element thrust falls with inflow through the
  disc; momentum theory adds the induced velocity, so climb costs thrust and
  forward flight gains translational lift. Rotation-induced inflow gives
  natural rotor damping.
- **Drag** per body axis, **wind** (steady plus Ornstein-Uhlenbeck gusts) and
  ground contact (a touchdown faster than the crash speed counts as a crash).
- **IMU**: white noise and randomly drifting biases on the gyro and the
  accelerometer, plus position and velocity measurement noise.

**Domain randomization** (`Randomization::default_for`), per episode:
- mass ±15–20% (passenger weight);
- inertia;
- off-centre mass;
- per-rotor thrust spread;
- motor time constants;
- wind and gusts;
- starting battery charge;
- IMU bias.

## Speed

`cargo run --release -p dronelab --example bench` (10-core Apple Silicon, 240 Hz physics, 30 Hz control):

| airframe | physics | envs / threads | env steps/s | physics steps/s |
|---|---|---|---|---|
| crazyflie_x | ideal | 8192 / 10 | 4.7 M | 37 M |
| crazyflie_x | full | 1 / 1 | 330 k | 2.6 M |
| crazyflie_x | full | 8192 / 10 | 2.2 M | 18 M |
| evtol_two_seat (16 rotors) | full | 1 / 1 | 200 k | 1.6 M |
| evtol_two_seat (16 rotors) | full | 1024 / 10 | 0.95 M | 7.6 M |

With Stable-Baselines3 PPO in the loop, `train_ppo.py` trains a single-seat
eVTOL reach policy (full realism and randomization) in 4M steps and 52 s.
Its return goes from 102 (holding still) to 210, ahead of the built-in PID
expert's 203, with no crashes. The policy update, not the simulator, is the
bottleneck there. A head-to-head against PyBullet
has not been run yet.

## Determinism

Each environment owns its random stream: xoshiro256++, with its seed
expanded by SplitMix64, plus `libm` math. So a seed reproduces an episode
bit for bit, whatever the thread count, batch size or machine. The tests
check this (`crates/sim/tests/batch.rs`). `state()["seed"]` tells you which
seed to replay.

## Python

```bash
python -m venv .venv && . .venv/bin/activate
pip install maturin numpy gymnasium stable-baselines3
cd crates/py && maturin develop --release
```

```python
import gymnasium as gym
import dronelab                      # registers dronelab/<Task>-<airframe>-v0

env = gym.make("dronelab/Reach-evtol_single_seat-v0", action="velocity")
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

Stable-Baselines3, unchanged, with the batch running in Rust:

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
from dronelab.vec_env import DronelabVecEnv

env = VecMonitor(DronelabVecEnv(num_envs=256, airframe="evtol_single_seat", action="velocity"))
PPO("MlpPolicy", env, n_steps=256, batch_size=8192, n_epochs=10,
    policy_kwargs=dict(log_std_init=-1.0)).learn(4_000_000)
```

See `crates/py/examples/train_ppo.py`. Heavy vehicles barely respond to
per-step action noise, so short rollouts (SB3's defaults) give PPO little
signal; use long rollouts and lower initial noise as above.

`BatchSim` is the raw batch API. It takes NumPy arrays in and returns them
out, and keyword arguments set:
- `airframe`;
- `task` (`"reach"`, `"race"`);
- `action` (`"rotor_thrust"`, `"ctbr"`, `"velocity"`);
- `physics_hz`, `control_hz`, `episode_seconds`;
- `realism` (`"full"`, `"ideal"` or a dict of flags);
- `randomize` (bool or a dict of ranges);
- `threads`, `track_json`.

`state()` returns pose, rotor speeds and thrusts, battery, IMU and wind for
logging. `expert_actions()` gives the PID baseline.

### Actions

- `rotor_thrust`: one value per rotor; 0 is hover, -1 is off, +1 is twice hover.
- `ctbr`: collective thrust plus body rates. The shared `RateControl` closes
  the rate loop at the physics rate.
- `velocity`: a world-frame velocity, flown by the shared `PidControl`.

### Observation

The observation is target minus position, velocity, rotation matrix, gyro and
rotor speeds, each scaled to about unit size. Race tasks add the next gate's
direction and the gate after it (see `crates/sim/src/lib.rs`).

## Racing

`task="race"` flies a circuit of gates scaled to the vehicle
(`dronelab.default_track_json(airframe)`; edit the JSON to make your own).
Gate 0 is start/finish; flying through every gate in order completes a lap,
and touching a frame is a crash. Lap times come back in `info["lap_time"]`.
`dronelab.Leaderboard(path)` keeps the fastest laps, with seeds, in a JSON
file.

```bash
cargo run --release -p dronelab --example race_pid
```

## Flight controller on STM32

`dronelab-core` is `no_std`, allocation-free `f32` code:

```bash
rustup target add thumbv7em-none-eabihf
cargo build -p dronelab-core --release --target thumbv7em-none-eabihf
```

```rust
use dronelab_core::{Control, KinematicState, PidControl, Target, VehicleParams, G};

let mut ctl = PidControl::new(VehicleParams::evtol_single_seat(), G).unwrap();
// in the 250-1000 Hz loop, with your state estimate:
let out = ctl.compute_control(dt, &state, &Target::at(setpoint));
for (i, t) in out.thrust().iter().enumerate() { /* newtons -> ESC command */ }
```

The controller is gym-pybullet-drones' DSLPIDControl structure. Its gains
are normalised by mass and inertia, so one tuning rule fits a 27 g quad and
a 620 kg eVTOL. Integrals carry anti-windup, and the mixer works for any
rotor layout, giving up yaw before roll and pitch when the rotors saturate.
`Control`, `BaseControl` and `set_pid_coefficients` mirror Python's
`BaseControl`.

## Tests

```bash
cargo test --release --workspace --exclude dronelab-py
.venv/bin/python -m pytest crates/py/tests
```

## Not done yet

- A benchmark against gym-pybullet-drones / PyBullet.
- Flight log export for ArduPilot or PX4 tools.
- Tilt-rotor or lift+cruise eVTOLs (only multicopters so far), vortex ring
  state, and ground effect.
- The PID racing baseline is slow and misses gates on the two-seat eVTOL;
  learned policies are expected to do the racing.

Bundled URDF files are from gym-pybullet-drones (MIT, see
`crates/sim/assets/LICENSE-gym-pybullet-drones`).

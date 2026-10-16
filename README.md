# dronelab

A fast, deterministic multirotor and eVTOL simulator for reinforcement
learning, written in Rust, after
[gym-pybullet-drones](https://github.com/learnsyslab/gym-pybullet-drones).
It comes with a Gymnasium / Stable-Baselines3 Python API, and its flight
controller is the same `no_std` code that runs on an STM32.

It has two layers:

- **`dronelab-compat`**, a verified port of gym-pybullet-drones v2.2.0
  (controllers, physics, environments). It is checked number by number
  against the upstream code, and upstream scripts run on it unchanged.
- **The eVTOL simulator** (`dronelab-core` + `dronelab`). It extends that
  work to any rotor layout, adds motor, battery and sensor realism, and runs
  thousands of vehicles in parallel.

```
crates/compat dronelab-compat  f64 line-by-line port of gym-pybullet-drones, golden-tested
crates/core   dronelab-core    no_std flight code: math, mixer, PID, BaseControl port
crates/sim    dronelab         simulator: dynamics, motors, battery, IMU, wind,
                               randomization, tasks, racing, rayon batch
crates/py     dronelab-py      pyo3 module + Python packages `dronelab`, `dronelab.compat`
```

## Faithful gym-pybullet-drones port

`dronelab.compat` has upstream's module layout, class names, constructor
arguments, attributes, spaces and return values. Change the import, or call
`install()` once and run existing scripts untouched:

```python
import dronelab.compat; dronelab.compat.install()
from gym_pybullet_drones.envs.HoverAviary import HoverAviary   # now runs in Rust
```

It has DSLPIDControl, CTBRControl, MRACControl, CtrlAviary, VelocityAviary,
HoverAviary and MultiHoverAviary (all five action types), all six `Physics`
modes and upstream's `Logger`. `CompatVecEnv` runs many HoverAviary or
MultiHoverAviary copies in parallel for SB3.

How it was checked. `crates/compat/tools/record_upstream.py` ran the pinned
upstream code (commit `0156312`) and recorded inputs and outputs; the Rust
tests compare against those recordings. The live tests run both packages
side by side, including upstream's own example scripts, unmodified.

| part | agreement with upstream |
|---|---|
| URDF constants (all 3 drones) | bit-identical |
| PyBullet / SciPy / transforms3d rotation helpers | within 1–3 rounding steps (≤ 7e-16) |
| DSLPIDControl, MRACControl, CTBRControl (100–150-step sequences) | ≤ 6.3e-15 relative |
| Forces and torques handed to PyBullet (rotors, ground effect, drag, downwash; 5,052 calls) | ≤ 1e-13 relative |
| `Physics.DYN` trajectories (2 drones, 300 substeps, each model) | ≤ 8.1e-15 relative |
| `PYB*` modes, up to 1 s of free flight | ≤ 1.1e-9 m, 6.3e-9 m/s, 2.7e-8 rad |
| HoverAviary / MultiHoverAviary / VelocityAviary episodes, DYN (11 cases) | identical observations (0.0) in 10; ≤ 3.6e-10 in one velocity-action case |
| same episodes, PYB (11 cases) | ≤ 5.8e-5 |
| upstream `pid.py` (PYB and DYN), `pid_velocity.py`, `downwash.py`, unmodified | within upstream's own drift (below) |

Some closed loops amplify rounding differences: velocity actions, and the
waypoint examples. There, upstream run with `KF` changed by one part in 10¹²
drifts away from itself too, by about 1 cm after 3 s in `pid.py`. The tests
measure that self-drift and require dronelab to stay within it (times 100
for DYN, times 1e4 for PYB, whose integrator stand-in starts further off),
instead of pretending such runs can match bit for bit. Episode steps past
that point are reported as chaotic and not compared.

What is not reproduced:
- PyBullet's ground-contact solver: contact is approximate, and the tests
  stop comparing at first touchdown;
- PyBullet's sample obstacles (warned about);
- the GUI, video recording and camera observations (these raise);
- `BetaAviary` (Betaflight SITL).

Upstream behaviour that is kept on purpose, because changing it would break
fidelity:
- `reset()` keeps PID integrals and the RL action buffer;
- RL environments use a CF2X controller even for CF2P;
- drag is applied along world axes;
- a time-limited episode lasts 242 steps;
- RL observations are float64 holding float32-rounded values, so Gymnasium's
  env checker rejects upstream and dronelab alike.

PYB modes are not bit-exact because they stand in for PyBullet's integrator.
Getting them this close took two details of how PyBullet applies link-frame
forces: the force direction comes from a cached link transform that is one
substep old, unless `getLinkStates(computeForwardKinematics=1)` has just
refreshed it; the lever arm comes from the current pose.

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

`Realism::FULL` (default) turns on everything; `Realism::IDEAL` turns the
extra effects off. Neither is upstream's model; for that, use
`dronelab-compat` above.

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
bottleneck there.

### Against gym-pybullet-drones (PyBullet)

`bench/pybullet_vs_dronelab.py` runs both simulators from Python on the
same task. The task is a Crazyflie (`CF2X` / `crazyflie_x`) hover with 240 Hz
physics, 30 Hz control (8 physics steps per env step) and automatic resets.
Each number is the median of three 3 s runs after an untimed warm-up, summed
over all environments. There are two action workloads:

- **random**: uniform random actions of matched size. HoverAviary's RPM
  action is ±5% of hover RPM, i.e. ±10% thrust, so dronelab's `rotor_thrust`
  actions are scaled to ±0.1. Episodes end every 10–30 steps, so reset cost
  is included.
- **hover**: the zero action. HoverAviary episodes run the full 8 s, so its
  number is close to pure stepping.

dronelab `ideal` has no randomization, like HoverAviary. `full` turns every
effect on, plus the default randomization. `SubprocVecEnv` runs one process
per env; dronelab uses one 10-thread rayon pool.

| simulator | physics | API | envs | random: env steps/s | random: physics steps/s | hover: env steps/s | hover: physics steps/s |
|---|---|---|---|---|---|---|---|
| gym-pybullet-drones | PYB | HoverAviary | 1 | 749 | 5.99 k | 5.26 k | 42.1 k |
| gym-pybullet-drones | PYB | SubprocVecEnv | 1 | 742 | 5.94 k | 3.89 k | 31.2 k |
| gym-pybullet-drones | PYB | SubprocVecEnv | 8 | 1.27 k | 10.2 k | 13.1 k | 105 k |
| gym-pybullet-drones | PYB | SubprocVecEnv | 10 | 1.31 k | 10.5 k | 13.6 k | 109 k |
| gym-pybullet-drones | DYN | HoverAviary | 1 | 576 | 4.61 k | 1.99 k | 15.9 k |
| gym-pybullet-drones | DYN | SubprocVecEnv | 1 | 636 | 5.09 k | 1.68 k | 13.4 k |
| gym-pybullet-drones | DYN | SubprocVecEnv | 8 | 1.23 k | 9.83 k | 6.36 k | 50.8 k |
| gym-pybullet-drones | DYN | SubprocVecEnv | 10 | 1.37 k | 11.0 k | 6.46 k | 51.7 k |
| dronelab | ideal | BatchSim | 1 | 349 k | 2.79 M | 349 k | 2.79 M |
| dronelab | ideal | DronelabVecEnv | 1 | 155 k | 1.24 M | 155 k | 1.24 M |
| dronelab | ideal | BatchSim | 1024 | 2.76 M | 22.1 M | 2.88 M | 23.0 M |
| dronelab | ideal | DronelabVecEnv | 1024 | 2.20 M | 17.6 M | 2.28 M | 18.3 M |
| dronelab | ideal | BatchSim | 8192 | 3.00 M | 24.0 M | 3.08 M | 24.6 M |
| dronelab | ideal | DronelabVecEnv | 8192 | 2.60 M | 20.8 M | 2.67 M | 21.4 M |
| dronelab | full | BatchSim | 1 | 190 k | 1.52 M | 191 k | 1.53 M |
| dronelab | full | DronelabVecEnv | 1 | 111 k | 891 k | 112 k | 892 k |
| dronelab | full | BatchSim | 1024 | 1.33 M | 10.7 M | 1.27 M | 10.1 M |
| dronelab | full | DronelabVecEnv | 1024 | 1.12 M | 8.99 M | 1.13 M | 9.01 M |
| dronelab | full | BatchSim | 8192 | 1.41 M | 11.3 M | 1.32 M | 10.5 M |
| dronelab | full | DronelabVecEnv | 8192 | 1.15 M | 9.19 M | 1.17 M | 9.34 M |

- **Through SB3 on all 10 cores**, the comparison is `DronelabVecEnv` with
  8192 envs against 10 HoverAviaries in `SubprocVecEnv`, taking the faster
  PyBullet mode. On hover, dronelab runs 197× as many env steps (ideal) or
  86× (full). With random actions it runs 1,900× (ideal) or 840× (full).
- **One env in one process** (`DronelabVecEnv(num_envs=1)` against a PYB
  HoverAviary): dronelab is 30× (ideal) or 21× (full) on hover, and 210× or
  150× with random actions.
- **Resets dominate PyBullet under random actions.** HoverAviary drops from
  5.26 k to 749 env steps/s, while episode ends go from 4 to 94 per 1000
  steps. Each of its resets calls `resetSimulation` and reloads the URDFs.
  dronelab resets inside `step` in Rust, so both of its workloads run at the
  same speed.
- `SubprocVecEnv` costs a single PYB env 26% (3.89 k against 5.26 k env
  steps/s), and 10 processes reach 2.6× one HoverAviary. DYN is slower than
  PYB because its dynamics run in Python.
- Two full runs agreed within 5%, except dronelab `full` at 1024 and 8192
  envs, which moved by up to 12%.

Measured on an Apple M5 (10 cores: 4 performance + 6 efficiency, 32 GB),
macOS 27.0.1, Python 3.12.8. Versions: pybullet 3.2.7 (built from source),
gym-pybullet-drones 2.2.0 (commit `0156312`), stable-baselines3 2.9.0,
NumPy 2.5.4 and dronelab 0.1.0. The raw numbers, with episode-end rates,
are in `bench/results.json`. To reproduce from the repo root:

```bash
uv venv --python 3.12 bench/.venv-pybullet
# pybullet has no macOS arm64 wheel; the flag gets its bundled zlib past
# the macOS SDK's fdopen() declaration. The build takes ~10 min.
CFLAGS="-Dfdopen=fdopen" VIRTUAL_ENV=bench/.venv-pybullet uv pip install \
  "gym-pybullet-drones @ git+https://github.com/learnsyslab/gym-pybullet-drones@01563120e060c9233a61f9cf00e6d4e5a8b331ab"
maturin build --release -m crates/py/Cargo.toml -o target/wheels
VIRTUAL_ENV=bench/.venv-pybullet uv pip install target/wheels/dronelab-*.whl
bench/.venv-pybullet/bin/python bench/pybullet_vs_dronelab.py --json bench/results.json
```

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
logging. `expert_actions()` gives the PID baseline. `record()` and
`save_log()` write flight logs (see below).

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

## Flight logs

A simulated flight can be saved as an ArduPilot DataFlash `.bin` or a PX4
ULog `.ulg`, so it opens in the same tools as your real flight logs.

```python
from dronelab import BatchSim

sim = BatchSim(num_envs=64, airframe="evtol_single_seat", action="velocity")
sim.record(env_index=0, origin=(47.4979, 19.0402, 110.0))  # lat, lon, alt AMSL
for _ in range(300):
    sim.step(sim.expert_actions())
sim.save_log("flight.bin")             # ArduPilot; format follows the extension
sim.save_log("flight.ulg")             # PX4
sim.stop_recording(0)
```

Recording runs every control step of one env, across episodes (with an
`episode N seed S` message at each start), until `stop_recording`. From Rust:
`env.start_log(GeoOrigin::default())`, then `env.stop_log()` and
`log.save(path, LogFormat::ULog)`; or run

```bash
cargo run --release -p dronelab --example export_log
```

| DataFlash | ULog | content |
|---|---|---|
| `ATT` | `vehicle_attitude` | attitude (no setpoint: none for rotor-thrust actions) |
| `IMU` | `sensor_combined`, `vehicle_angular_velocity` | noisy gyro and accelerometer |
| `GPS`, `POS` | `vehicle_global_position` | latitude, longitude, altitude AMSL |
| `XKF1` | `vehicle_local_position` | NED position and velocity (true state) |
| `BAT` | `battery_status` | voltage, current, mAh and Wh used, charge |
| `RCOU`, `RCO2`, `RCO3` | `actuator_outputs` | rotor commands as PWM |
| `ESC` | | rotor RPM, one message per rotor |
| `PARM`, `MSG` | parameters, info, logged messages | pack size, frame, episode starts |

Conventions: dronelab's world is taken as ENU (x east, y north, z up) and its
body frame is FLU; the logs are NED / FRD, so north = y, east = x,
down = -z, and a dronelab yaw of 0 is a heading of 90°. Pitch changes sign
(dronelab's +pitch about +y-left is nose down); roll keeps it. Rotor `i`
is output `i + 1`, with PWM = 1000 + 1000 × commanded fraction of max thrust.
Latitude and longitude are a flat-earth projection around `origin`
(default: ArduPilot SITL's home).

Open them with:

```bash
pip install pymavlink MAVProxy pyulog
MAVExplorer.py flight.bin                            # then: graph ATT.Roll ATT.Pitch, map
mavlogdump.py --types BAT,ATT flight.bin             # text dump
ulog_info flight.ulg                                 # topics and counts
ulog2csv flight.ulg                                  # one CSV per topic
```

- UAV Log Viewer: drop `flight.bin` on <https://plot.ardupilot.org>.
- PX4 Flight Review: upload `flight.ulg` at <https://review.px4.io>
  (choose "keep private"), or run it locally from
  <https://github.com/PX4/flight_review>.
- PlotJuggler opens both formats.

`crates/py/tests/test_logs.py` reads the logs back with `pyulog` and
pymavlink's `DFReader` and checks message counts, altitude, voltage and
attitude against `state()`.

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
.venv/bin/python -m pytest crates/py/tests   # log tests need: pip install pyulog pymavlink
# side by side with real gym-pybullet-drones (needs PyBullet; see bench/):
bench/.venv-pybullet/bin/python -m pytest crates/py/tests/test_compat_live.py
# re-record upstream reference data:
bench/.venv-pybullet/bin/python crates/compat/tools/record_upstream.py
```

## Not done yet

- In `dronelab.compat`: PyBullet's ground contact, sample obstacles, GUI,
  cameras and `BetaAviary`.
- Tilt-rotor or lift+cruise eVTOLs (only multicopters so far), vortex ring
  state, and ground effect.
- The PID racing baseline is slow and misses gates on the two-seat eVTOL;
  learned policies are expected to do the racing.

Bundled URDF files are from gym-pybullet-drones (MIT, see
`crates/sim/assets/LICENSE-gym-pybullet-drones`).

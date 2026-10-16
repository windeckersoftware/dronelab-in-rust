//! dronelab: a fast, deterministic multirotor and eVTOL simulator for
//! reinforcement learning, after [gym-pybullet-drones](https://github.com/learnsyslab/gym-pybullet-drones).
//!
//! - [`BatchEnv`] steps thousands of environments in parallel with rayon.
//! - A seed fixes an episode completely, on any machine and thread count.
//! - Motor lag, battery sag, prop thrust curves, drag, wind and a noisy,
//!   drifting IMU ([`Realism`]); per-episode [`Randomization`].
//! - The controller and mixer come from `dronelab-core`, the same `no_std`
//!   code that runs on the flight controller.
//! - Racing with gates, lap timing and a leaderboard ([`track`]).
//!
//! # Observation layout
//!
//! | index | content |
//! |---|---|
//! | 0..3 | target (or next gate) minus position, / `pos_scale` |
//! | 3..6 | velocity, / `vel_scale` |
//! | 6..15 | rotation matrix, row-major |
//! | 15..18 | gyro body rates, / `max_rate` |
//! | 18..18+n | rotor speeds, / `omega_max` |
//! | race only: +3 | next gate's direction of travel |
//! | race only: +3 | the gate after next minus position, / `pos_scale` |

pub mod airframe;
pub mod batch;
pub mod battery;
pub mod dynamics;
pub mod enums;
pub mod env;
pub mod randomization;
pub mod rng;
pub mod sensors;
pub mod track;
pub mod urdf;

pub use airframe::Airframe;
pub use batch::BatchEnv;
pub use dronelab_core as core;
pub use dynamics::Realism;
pub use enums::DroneModel;
pub use env::{ActionMode, Env, EnvConfig, StepInfo, Task};
pub use randomization::Randomization;
pub use track::{Gate, LapRecord, Leaderboard, Track};
pub use urdf::{UrdfError, UrdfParam};

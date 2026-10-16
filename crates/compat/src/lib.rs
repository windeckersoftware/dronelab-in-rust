//! dronelab-compat: a line-by-line `f64` port of
//! [gym-pybullet-drones](https://github.com/learnsyslab/gym-pybullet-drones)
//! (v2.2.0, commit `0156312`).
//!
//! Every formula follows the upstream source, quirks included, and the tests
//! compare against outputs recorded from upstream itself
//! (`tools/record_upstream.py`, `tests/golden/`). See the crate README for
//! what matches exactly and what only approximately.

// Index loops mirror the NumPy expressions they port.
#![allow(clippy::needless_range_loop)]

pub mod aviary;
pub mod batch;
pub mod control;
pub mod math;
pub mod mrac_tables;
pub mod params;
pub mod physics;

pub use aviary::{ActionType, Aviary, AviaryConfig, EnvKind, StepResult};
pub use batch::AviaryBatch;
pub use dronelab::DroneModel;
pub use params::{G, Params};
pub use physics::Physics;

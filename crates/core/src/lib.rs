//! Flight code shared by the dronelab simulator and the real flight controller.
//!
//! Everything here is `no_std`, allocation-free and uses `f32`, so the same
//! controller and mixer run in the simulator and on an STM32 (Cortex-M4F/M7).
//! Check with `cargo build -p dronelab-core --target thumbv7em-none-eabihf`.

#![no_std]

pub mod control;
pub mod math;
pub mod mixer;
pub mod pid;
pub mod vehicle;

pub use control::{BaseControl, Control, ControlError, ControlOutput, KinematicState, Target};
pub use math::{Mat3, Quat, Vec3};
pub use mixer::Mixer;
pub use pid::{PidCoefficients, PidControl, PidGains, RateControl};
pub use vehicle::{Rotor, VehicleParams};

/// Largest rotor count a vehicle may have. Arrays are sized by it so that no
/// heap is needed; large multicopter eVTOLs have up to 18 rotors.
pub const MAX_ROTORS: usize = 24;

/// Standard gravity, m/s^2.
pub const G: f32 = 9.81;

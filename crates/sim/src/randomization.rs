//! Domain randomization: a different true vehicle and weather every episode,
//! so that a learned policy does not overfit one exact model.

use dronelab_core::math::{Vec3, cos, sin};

use crate::airframe::Airframe;
use crate::dynamics::Physical;
use crate::rng::Rng;
use crate::sensors::Wind;

/// Ranges `[lo, hi]` sampled uniformly at every reset.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Randomization {
    /// Total mass scale (payload, passenger weight).
    pub mass_scale: [f32; 2],
    pub inertia_scale: [f32; 2],
    /// Largest horizontal centre-of-mass offset, as a fraction of the
    /// airframe radius (passenger seated off-centre, cargo).
    pub com_offset: f32,
    /// Per-rotor thrust scale (prop wear, manufacturing spread).
    pub motor_strength: [f32; 2],
    /// Motor time-constant scale.
    pub motor_tau_scale: [f32; 2],
    /// Mean horizontal wind speed, m/s, from a random direction.
    pub wind_speed: [f32; 2],
    /// Gust standard deviation, m/s.
    pub gust_std: [f32; 2],
    /// Initial battery state of charge.
    pub battery_soc: [f32; 2],
    /// Scale on the IMU turn-on bias.
    pub imu_bias_scale: f32,
}

impl Randomization {
    /// The nominal vehicle in calm air with a full battery.
    pub const NONE: Randomization = Randomization {
        mass_scale: [1.0, 1.0],
        inertia_scale: [1.0, 1.0],
        com_offset: 0.0,
        motor_strength: [1.0, 1.0],
        motor_tau_scale: [1.0, 1.0],
        wind_speed: [0.0, 0.0],
        gust_std: [0.0, 0.0],
        battery_soc: [1.0, 1.0],
        imu_bias_scale: 1.0,
    };

    /// Moderate ranges, with wind scaled to what the vehicle can fly in.
    pub fn default_for(af: &Airframe) -> Self {
        let wind = 0.4 * af.vehicle.max_speed_xy;
        Randomization {
            mass_scale: [0.85, 1.2],
            inertia_scale: [0.85, 1.2],
            com_offset: 0.03,
            motor_strength: [0.9, 1.05],
            motor_tau_scale: [0.8, 1.3],
            wind_speed: [0.0, wind],
            gust_std: [0.0, 0.25 * wind],
            battery_soc: [0.6, 1.0],
            imu_bias_scale: 1.0,
        }
    }

    /// Draws the true vehicle, the wind, and the initial charge.
    pub fn sample(&self, af: &Airframe, rng: &mut Rng) -> (Physical, Wind, f32) {
        let mut r = |[lo, hi]: [f32; 2]| if lo == hi { lo } else { rng.range(lo, hi) };
        let mut p = Physical::nominal(af);
        p.mass *= r(self.mass_scale);
        p.inertia = p.inertia * r(self.inertia_scale);
        let tau = r(self.motor_tau_scale);
        p.motor_tau_up *= tau;
        p.motor_tau_down *= tau;
        for i in 0..af.vehicle.rotor_count() {
            let k = r(self.motor_strength);
            p.kf[i] *= k;
            p.km[i] *= k;
        }
        let speed = r(self.wind_speed);
        let gust_std = r(self.gust_std);
        let soc = r(self.battery_soc);
        if self.com_offset > 0.0 {
            let (d, a) = (
                rng.range(0.0, self.com_offset * af.radius()),
                rng.range(0.0, 2.0 * core::f32::consts::PI),
            );
            p.com = Vec3::new(d * cos(a), d * sin(a), 0.0);
        }
        let heading = rng.range(0.0, 2.0 * core::f32::consts::PI);
        let wind = Wind {
            mean: Vec3::new(speed * cos(heading), speed * sin(heading), 0.0),
            gust: Vec3::ZERO,
            gust_std,
            gust_tau: 2.0,
        };
        (p, wind, soc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn none_gives_nominal_vehicle() {
        let af = Airframe::evtol_single_seat();
        let (p, w, soc) = Randomization::NONE.sample(&af, &mut Rng::new(1));
        assert_eq!(p, Physical::nominal(&af));
        assert_eq!(
            w,
            Wind {
                gust_tau: 2.0,
                ..Wind::CALM
            }
        );
        assert_eq!(soc, 1.0);
    }

    #[test]
    fn samples_stay_in_range_and_vary() {
        let af = Airframe::evtol_two_seat();
        let rz = Randomization::default_for(&af);
        let mut rng = Rng::new(2);
        let masses: Vec<f32> = (0..50).map(|_| rz.sample(&af, &mut rng).0.mass).collect();
        let m = af.vehicle.mass;
        assert!(masses.iter().all(|&x| x >= 0.85 * m && x <= 1.2 * m));
        assert!(masses.iter().any(|&x| x != masses[0]));
    }
}

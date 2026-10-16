//! IMU with white noise and drifting biases, and the wind field.

use dronelab_core::math::Vec3;

use crate::rng::Rng;

fn normal3(rng: &mut Rng) -> Vec3 {
    Vec3::new(rng.normal(), rng.normal(), rng.normal())
}

/// Noise figures of a MEMS IMU. Defaults are typical of a flight-controller
/// grade part (BMI088/ICM-42688 class).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImuSpec {
    /// White noise per sample, rad/s.
    pub gyro_noise: f32,
    /// Standard deviation of the turn-on bias, rad/s.
    pub gyro_bias: f32,
    /// Bias random walk, rad/s per sqrt(s).
    pub gyro_walk: f32,
    /// White noise per sample, m/s^2.
    pub accel_noise: f32,
    pub accel_bias: f32,
    pub accel_walk: f32,
}

impl Default for ImuSpec {
    fn default() -> Self {
        ImuSpec {
            gyro_noise: 0.005,
            gyro_bias: 0.01,
            gyro_walk: 0.0005,
            accel_noise: 0.05,
            accel_bias: 0.05,
            accel_walk: 0.002,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ImuReading {
    /// Body rates, rad/s.
    pub gyro: Vec3,
    /// Specific force in the body frame, m/s^2 (reads +g upward at rest).
    pub accel: Vec3,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Imu {
    pub spec: ImuSpec,
    pub gyro_bias: Vec3,
    pub accel_bias: Vec3,
    pub last: ImuReading,
}

impl Imu {
    /// A fresh IMU with turn-on biases drawn from `rng`, scaled by `bias_scale`.
    pub fn new(spec: ImuSpec, bias_scale: f32, rng: &mut Rng) -> Self {
        Imu {
            spec,
            gyro_bias: normal3(rng) * (spec.gyro_bias * bias_scale),
            accel_bias: normal3(rng) * (spec.accel_bias * bias_scale),
            last: ImuReading::default(),
        }
    }

    /// A noiseless IMU.
    pub fn ideal() -> Self {
        let spec = ImuSpec {
            gyro_noise: 0.0,
            gyro_bias: 0.0,
            gyro_walk: 0.0,
            accel_noise: 0.0,
            accel_bias: 0.0,
            accel_walk: 0.0,
        };
        Imu {
            spec,
            gyro_bias: Vec3::ZERO,
            accel_bias: Vec3::ZERO,
            last: ImuReading::default(),
        }
    }

    fn is_ideal(&self) -> bool {
        self.spec.gyro_noise == 0.0
            && self.spec.gyro_walk == 0.0
            && self.spec.accel_noise == 0.0
            && self.spec.accel_walk == 0.0
    }

    pub fn sample(
        &mut self,
        ang_vel: Vec3,
        specific_force: Vec3,
        dt: f32,
        rng: &mut Rng,
    ) -> ImuReading {
        if self.is_ideal() {
            self.last = ImuReading {
                gyro: ang_vel + self.gyro_bias,
                accel: specific_force + self.accel_bias,
            };
            return self.last;
        }
        let s = &self.spec;
        let root_dt = libm::sqrtf(dt);
        self.gyro_bias += normal3(rng) * (s.gyro_walk * root_dt);
        self.accel_bias += normal3(rng) * (s.accel_walk * root_dt);
        self.last = ImuReading {
            gyro: ang_vel + self.gyro_bias + normal3(rng) * s.gyro_noise,
            accel: specific_force + self.accel_bias + normal3(rng) * s.accel_noise,
        };
        self.last
    }
}

/// Steady wind plus Ornstein-Uhlenbeck gusts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Wind {
    pub mean: Vec3,
    pub gust: Vec3,
    /// Gust standard deviation, m/s (vertical gusts are a third of this).
    pub gust_std: f32,
    /// Gust correlation time, s.
    pub gust_tau: f32,
}

impl Wind {
    pub const CALM: Wind = Wind {
        mean: Vec3::ZERO,
        gust: Vec3::ZERO,
        gust_std: 0.0,
        gust_tau: 2.0,
    };

    pub fn velocity(&self) -> Vec3 {
        self.mean + self.gust
    }

    pub fn step(&mut self, dt: f32, rng: &mut Rng) {
        if self.gust_std == 0.0 {
            return;
        }
        let a = libm::expf(-dt / self.gust_tau);
        let b = self.gust_std * libm::sqrtf(1.0 - a * a);
        let n = normal3(rng);
        self.gust = self.gust * a + Vec3::new(n.x * b, n.y * b, n.z * b / 3.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gyro_bias_drifts() {
        let mut rng = Rng::new(3);
        let mut imu = Imu::new(ImuSpec::default(), 1.0, &mut rng);
        let b0 = imu.gyro_bias;
        for _ in 0..10_000 {
            imu.sample(Vec3::ZERO, Vec3::Z * 9.81, 0.004, &mut rng);
        }
        assert!((imu.gyro_bias - b0).norm() > 1e-4);
    }

    #[test]
    fn gusts_have_requested_spread() {
        let mut rng = Rng::new(4);
        let mut w = Wind {
            gust_std: 3.0,
            ..Wind::CALM
        };
        let mut sq = 0.0;
        let n = 50_000;
        for _ in 0..n {
            w.step(0.05, &mut rng);
            sq += w.gust.x * w.gust.x;
        }
        let std = libm::sqrtf(sq / n as f32);
        assert!((std - 3.0).abs() < 0.5, "{std}");
    }
}

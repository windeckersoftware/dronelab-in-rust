//! Control allocation: total thrust and body torques to per-rotor thrust.
//!
//! Works for any rotor layout (quad, X8 coaxial, 16-rotor eVTOL) by inverting
//! the allocation matrix once at start-up.

use crate::MAX_ROTORS;
use crate::math::Vec3;
use crate::vehicle::VehicleParams;

/// Maps `[thrust, torque_x, torque_y, torque_z]` to rotor thrusts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mixer {
    /// Pseudo-inverse of the allocation matrix, one row per rotor.
    pinv: [[f32; 4]; MAX_ROTORS],
    max_thrust: [f32; MAX_ROTORS],
    rotor_count: usize,
}

/// Result of [`Mixer::mix`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MixOutput {
    /// Per-rotor thrust, N. Only the first `rotor_count` entries are used.
    pub thrust: [f32; MAX_ROTORS],
    /// Some request could not be met within the rotors' limits.
    pub saturated: bool,
}

impl Mixer {
    /// Returns `None` if the layout cannot produce independent roll, pitch
    /// and yaw torques (singular allocation matrix).
    pub fn new(vehicle: &VehicleParams) -> Option<Self> {
        let rotors = vehicle.rotors();
        // Allocation matrix A (4 x n): wrench = A * thrusts.
        let mut a = [[0.0f32; MAX_ROTORS]; 4];
        for (i, r) in rotors.iter().enumerate() {
            a[0][i] = 1.0;
            a[1][i] = r.position.y;
            a[2][i] = -r.position.x;
            a[3][i] = -r.spin * r.torque_ratio;
        }
        // pinv = A^T (A A^T)^-1
        let mut aat = [[0.0f32; 4]; 4];
        for (i, row) in aat.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell = (0..rotors.len()).map(|k| a[i][k] * a[j][k]).sum();
            }
        }
        let inv = invert4(aat)?;
        let mut pinv = [[0.0f32; 4]; MAX_ROTORS];
        let mut max_thrust = [0.0f32; MAX_ROTORS];
        for k in 0..rotors.len() {
            for j in 0..4 {
                pinv[k][j] = (0..4).map(|i| a[i][k] * inv[i][j]).sum();
            }
            max_thrust[k] = rotors[k].max_thrust;
        }
        Some(Mixer {
            pinv,
            max_thrust,
            rotor_count: rotors.len(),
        })
    }

    pub fn rotor_count(&self) -> usize {
        self.rotor_count
    }

    /// Splits the wrench across rotors, keeping each within `[0, max_thrust]`.
    ///
    /// When the request does not fit, yaw is given up first, then collective
    /// thrust, so roll and pitch control is kept as long as possible.
    pub fn mix(&self, thrust: f32, torque: Vec3) -> MixOutput {
        let n = self.rotor_count;
        let mut t = [0.0f32; MAX_ROTORS];
        let mut yaw = [0.0f32; MAX_ROTORS];
        let mut col = [0.0f32; MAX_ROTORS];
        for k in 0..n {
            let p = &self.pinv[k];
            t[k] = p[0] * thrust + p[1] * torque.x + p[2] * torque.y;
            yaw[k] = p[3] * torque.z;
            col[k] = p[0];
        }
        let mut saturated = false;

        // Shift collective so roll/pitch fit, if possible.
        let shift = self.fit_along(&t, &col, &mut saturated);
        for k in 0..n {
            t[k] += shift * col[k];
        }

        // Add as much of the yaw request as fits.
        let mut scale = 1.0f32;
        for k in 0..n {
            let (lo, hi) = (0.0 - t[k], self.max_thrust[k] - t[k]);
            if yaw[k] > hi && yaw[k] > 0.0 {
                scale = scale.min(hi.max(0.0) / yaw[k]);
            } else if yaw[k] < lo && yaw[k] < 0.0 {
                scale = scale.min(lo.min(0.0) / yaw[k]);
            }
        }
        if scale < 1.0 {
            saturated = true;
        }
        for k in 0..n {
            t[k] += scale * yaw[k];
        }

        // Anything still out of range is clipped.
        for (tk, &max) in t[..n].iter_mut().zip(&self.max_thrust) {
            let clipped = tk.clamp(0.0, max);
            if clipped != *tk {
                saturated = true;
                *tk = clipped;
            }
        }
        MixOutput {
            thrust: t,
            saturated,
        }
    }

    /// The shift `s` closest to zero such that `t + s * dir` is within limits,
    /// or the best compromise if no such shift exists.
    fn fit_along(
        &self,
        t: &[f32; MAX_ROTORS],
        dir: &[f32; MAX_ROTORS],
        saturated: &mut bool,
    ) -> f32 {
        let (mut lo, mut hi) = (f32::NEG_INFINITY, f32::INFINITY);
        for k in 0..self.rotor_count {
            let d = dir[k];
            if d.abs() < 1e-9 {
                continue;
            }
            let (a, b) = ((0.0 - t[k]) / d, (self.max_thrust[k] - t[k]) / d);
            let (a, b) = if d > 0.0 { (a, b) } else { (b, a) };
            lo = lo.max(a);
            hi = hi.min(b);
        }
        if lo <= hi {
            let s = 0.0f32.clamp(lo, hi);
            if s != 0.0 {
                *saturated = true;
            }
            s
        } else {
            *saturated = true;
            0.5 * (lo + hi)
        }
    }

    /// The wrench a set of rotor thrusts produces (A * thrusts).
    pub fn wrench(vehicle: &VehicleParams, thrust: &[f32]) -> (f32, Vec3) {
        let mut total = 0.0;
        let mut tau = Vec3::ZERO;
        for (r, &t) in vehicle.rotors().iter().zip(thrust) {
            total += t;
            tau += Vec3::new(
                r.position.y * t,
                -r.position.x * t,
                -r.spin * r.torque_ratio * t,
            );
        }
        (total, tau)
    }
}

/// Inverts a 4x4 matrix by Gauss-Jordan elimination with partial pivoting.
fn invert4(mut m: [[f32; 4]; 4]) -> Option<[[f32; 4]; 4]> {
    let mut inv = [[0.0f32; 4]; 4];
    for (i, row) in inv.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    let scale = m.iter().flatten().fold(0.0f32, |a, &v| a.max(v.abs()));
    for c in 0..4 {
        let p = (c..4).max_by(|&a, &b| m[a][c].abs().total_cmp(&m[b][c].abs()))?;
        if m[p][c].abs() <= 1e-7 * scale {
            return None;
        }
        m.swap(c, p);
        inv.swap(c, p);
        let d = 1.0 / m[c][c];
        for j in 0..4 {
            m[c][j] *= d;
            inv[c][j] *= d;
        }
        for r in 0..4 {
            if r != c {
                let f = m[r][c];
                for j in 0..4 {
                    m[r][j] -= f * m[c][j];
                    inv[r][j] -= f * inv[c][j];
                }
            }
        }
    }
    Some(inv)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_round_trip(v: &VehicleParams) {
        let mixer = Mixer::new(v).unwrap();
        let hover = v.weight();
        let torque = Vec3::new(0.01, -0.02, 0.005) * (v.mass * 0.1);
        let out = mixer.mix(hover, torque);
        assert!(!out.saturated, "{}", v.name);
        let (t, tau) = Mixer::wrench(v, &out.thrust[..v.rotor_count()]);
        assert!(
            (t - hover).abs() < 1e-3 * hover,
            "{}: thrust {t} vs {hover}",
            v.name
        );
        assert!(
            (tau - torque).norm() < 1e-3 * hover,
            "{}: torque {tau:?}",
            v.name
        );
    }

    #[test]
    fn mixes_every_preset_exactly_when_unsaturated() {
        for name in VehicleParams::PRESETS {
            check_round_trip(&VehicleParams::preset(name).unwrap());
        }
    }

    #[test]
    fn saturation_drops_yaw_before_roll() {
        let v = VehicleParams::evtol_single_seat();
        let mixer = Mixer::new(&v).unwrap();
        let roll = 400.0;
        let out = mixer.mix(v.weight(), Vec3::new(roll, 0.0, 1.0e4));
        assert!(out.saturated);
        let (_, tau) = Mixer::wrench(&v, &out.thrust[..8]);
        assert!((tau.x - roll).abs() < 1.0, "roll kept: {tau:?}");
        for (k, t) in out.thrust[..8].iter().enumerate() {
            assert!(*t >= 0.0 && *t <= v.rotors()[k].max_thrust);
        }
    }

    #[test]
    fn rejects_layout_without_yaw_authority() {
        let mut v = VehicleParams::crazyflie_x();
        for r in v.rotors_mut() {
            r.torque_ratio = 0.0;
        }
        assert!(Mixer::new(&v).is_none());
    }
}

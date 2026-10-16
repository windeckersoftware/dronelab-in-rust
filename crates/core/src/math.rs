//! Minimal 3D math: vectors, 3x3 matrices and unit quaternions.
//!
//! Conventions: world frame is z-up; quaternions rotate body to world and are
//! stored scalar-first (`w, x, y, z`). PyBullet uses `x, y, z, w`; see
//! [`Quat::from_xyzw`].

use core::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};

#[inline]
pub fn sqrt(x: f32) -> f32 {
    libm::sqrtf(x)
}
#[inline]
pub fn sin(x: f32) -> f32 {
    libm::sinf(x)
}
#[inline]
pub fn cos(x: f32) -> f32 {
    libm::cosf(x)
}
#[inline]
pub fn tan(x: f32) -> f32 {
    libm::tanf(x)
}
#[inline]
pub fn atan2(y: f32, x: f32) -> f32 {
    libm::atan2f(y, x)
}
#[inline]
pub fn asin(x: f32) -> f32 {
    libm::asinf(x.clamp(-1.0, 1.0))
}
#[inline]
pub fn exp(x: f32) -> f32 {
    libm::expf(x)
}
#[inline]
pub fn tanh(x: f32) -> f32 {
    libm::tanhf(x)
}

/// Wraps an angle to (-pi, pi].
pub fn wrap_angle(a: f32) -> f32 {
    atan2(sin(a), cos(a))
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Vec3 = Vec3::new(0.0, 0.0, 0.0);
    pub const X: Vec3 = Vec3::new(1.0, 0.0, 0.0);
    pub const Y: Vec3 = Vec3::new(0.0, 1.0, 0.0);
    pub const Z: Vec3 = Vec3::new(0.0, 0.0, 1.0);

    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Vec3 { x, y, z }
    }

    pub const fn splat(v: f32) -> Self {
        Vec3::new(v, v, v)
    }

    pub const fn from_array(a: [f32; 3]) -> Self {
        Vec3::new(a[0], a[1], a[2])
    }

    pub const fn to_array(self) -> [f32; 3] {
        [self.x, self.y, self.z]
    }

    pub fn dot(self, o: Vec3) -> f32 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    pub fn cross(self, o: Vec3) -> Vec3 {
        Vec3::new(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }

    /// Component-wise product.
    pub fn mul_elem(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x * o.x, self.y * o.y, self.z * o.z)
    }

    pub fn norm_squared(self) -> f32 {
        self.dot(self)
    }

    pub fn norm(self) -> f32 {
        sqrt(self.norm_squared())
    }

    /// Unit vector in the same direction, or `fallback` if `self` is ~zero.
    pub fn normalized_or(self, fallback: Vec3) -> Vec3 {
        let n = self.norm();
        if n > 1e-9 { self * (1.0 / n) } else { fallback }
    }

    /// Component-wise clamp to `[-limit, limit]`.
    pub fn clamp_abs(self, limit: Vec3) -> Vec3 {
        Vec3::new(
            self.x.clamp(-limit.x, limit.x),
            self.y.clamp(-limit.y, limit.y),
            self.z.clamp(-limit.z, limit.z),
        )
    }
}

impl Add for Vec3 {
    type Output = Vec3;
    fn add(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }
}
impl AddAssign for Vec3 {
    fn add_assign(&mut self, o: Vec3) {
        *self = *self + o;
    }
}
impl Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }
}
impl SubAssign for Vec3 {
    fn sub_assign(&mut self, o: Vec3) {
        *self = *self - o;
    }
}
impl Mul<f32> for Vec3 {
    type Output = Vec3;
    fn mul(self, s: f32) -> Vec3 {
        Vec3::new(self.x * s, self.y * s, self.z * s)
    }
}
impl Neg for Vec3 {
    type Output = Vec3;
    fn neg(self) -> Vec3 {
        Vec3::new(-self.x, -self.y, -self.z)
    }
}

/// Row-major 3x3 matrix.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat3 {
    pub m: [[f32; 3]; 3],
}

impl Mat3 {
    pub const IDENTITY: Mat3 = Mat3 {
        m: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    };

    pub fn from_cols(c0: Vec3, c1: Vec3, c2: Vec3) -> Mat3 {
        Mat3 {
            m: [[c0.x, c1.x, c2.x], [c0.y, c1.y, c2.y], [c0.z, c1.z, c2.z]],
        }
    }

    pub fn col(&self, i: usize) -> Vec3 {
        Vec3::new(self.m[0][i], self.m[1][i], self.m[2][i])
    }

    pub fn transpose(&self) -> Mat3 {
        let m = &self.m;
        Mat3 {
            m: [
                [m[0][0], m[1][0], m[2][0]],
                [m[0][1], m[1][1], m[2][1]],
                [m[0][2], m[1][2], m[2][2]],
            ],
        }
    }

    pub fn mul_vec(&self, v: Vec3) -> Vec3 {
        let m = &self.m;
        Vec3::new(
            m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
            m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
            m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z,
        )
    }

    pub fn mul_mat(&self, o: &Mat3) -> Mat3 {
        let mut r = [[0.0; 3]; 3];
        for (i, row) in r.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell = (0..3).map(|k| self.m[i][k] * o.m[k][j]).sum();
            }
        }
        Mat3 { m: r }
    }

    pub fn sub_mat(&self, o: &Mat3) -> Mat3 {
        let mut r = self.m;
        for (i, row) in r.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell -= o.m[i][j];
            }
        }
        Mat3 { m: r }
    }

    /// The vector of a skew-symmetric matrix (inverse of the hat map).
    pub fn vee(&self) -> Vec3 {
        Vec3::new(self.m[2][1], self.m[0][2], self.m[1][0])
    }

    /// Row-major flat copy, handy for observations.
    pub fn to_array(&self) -> [f32; 9] {
        let m = &self.m;
        [
            m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2],
        ]
    }
}

/// Unit quaternion rotating body-frame vectors into the world frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quat {
    pub w: f32,
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Default for Quat {
    fn default() -> Self {
        Quat::IDENTITY
    }
}

impl Quat {
    pub const IDENTITY: Quat = Quat {
        w: 1.0,
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    pub const fn new(w: f32, x: f32, y: f32, z: f32) -> Self {
        Quat { w, x, y, z }
    }

    /// From PyBullet's `[x, y, z, w]` order.
    pub const fn from_xyzw(q: [f32; 4]) -> Self {
        Quat::new(q[3], q[0], q[1], q[2])
    }

    /// To PyBullet's `[x, y, z, w]` order.
    pub const fn to_xyzw(self) -> [f32; 4] {
        [self.x, self.y, self.z, self.w]
    }

    /// From roll, pitch, yaw (intrinsic Z-Y-X, the aerospace convention).
    pub fn from_rpy(roll: f32, pitch: f32, yaw: f32) -> Self {
        let (sr, cr) = (sin(roll * 0.5), cos(roll * 0.5));
        let (sp, cp) = (sin(pitch * 0.5), cos(pitch * 0.5));
        let (sy, cy) = (sin(yaw * 0.5), cos(yaw * 0.5));
        Quat::new(
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        )
    }

    /// Roll, pitch, yaw (intrinsic Z-Y-X).
    pub fn to_rpy(self) -> Vec3 {
        let Quat { w, x, y, z } = self;
        let roll = atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y));
        let pitch = asin(2.0 * (w * y - z * x));
        let yaw = atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
        Vec3::new(roll, pitch, yaw)
    }

    /// Rotation of `angle` radians about the unit vector `axis`.
    pub fn from_axis_angle(axis: Vec3, angle: f32) -> Self {
        let s = sin(angle * 0.5);
        Quat::new(cos(angle * 0.5), axis.x * s, axis.y * s, axis.z * s)
    }

    pub fn normalized(self) -> Self {
        let n = sqrt(self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z);
        if n < 1e-9 {
            return Quat::IDENTITY;
        }
        let k = 1.0 / n;
        Quat::new(self.w * k, self.x * k, self.y * k, self.z * k)
    }

    pub fn conjugate(self) -> Self {
        Quat::new(self.w, -self.x, -self.y, -self.z)
    }

    /// Hamilton product `self * o`.
    pub fn mul_quat(self, o: Quat) -> Quat {
        Quat::new(
            self.w * o.w - self.x * o.x - self.y * o.y - self.z * o.z,
            self.w * o.x + self.x * o.w + self.y * o.z - self.z * o.y,
            self.w * o.y - self.x * o.z + self.y * o.w + self.z * o.x,
            self.w * o.z + self.x * o.y - self.y * o.x + self.z * o.w,
        )
    }

    /// Rotates a body-frame vector into the world frame.
    pub fn rotate(self, v: Vec3) -> Vec3 {
        let u = Vec3::new(self.x, self.y, self.z);
        let t = u.cross(v) * 2.0;
        v + t * self.w + u.cross(t)
    }

    /// Rotates a world-frame vector into the body frame.
    pub fn inverse_rotate(self, v: Vec3) -> Vec3 {
        self.conjugate().rotate(v)
    }

    /// Rotation matrix whose columns are the body axes in world coordinates.
    pub fn to_mat(self) -> Mat3 {
        let Quat { w, x, y, z } = self;
        Mat3 {
            m: [
                [
                    1.0 - 2.0 * (y * y + z * z),
                    2.0 * (x * y - w * z),
                    2.0 * (x * z + w * y),
                ],
                [
                    2.0 * (x * y + w * z),
                    1.0 - 2.0 * (x * x + z * z),
                    2.0 * (y * z - w * x),
                ],
                [
                    2.0 * (x * z - w * y),
                    2.0 * (y * z + w * x),
                    1.0 - 2.0 * (x * x + y * y),
                ],
            ],
        }
    }

    /// Advances the orientation by body angular velocity `omega` over `dt`,
    /// using the exact exponential map for constant `omega`.
    pub fn integrate(self, omega: Vec3, dt: f32) -> Quat {
        let angle = omega.norm() * dt;
        if angle < 1e-9 {
            return self;
        }
        let dq = Quat::from_axis_angle(omega * (1.0 / omega.norm()), angle);
        self.mul_quat(dq).normalized()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: Vec3, b: Vec3) -> bool {
        (a - b).norm() < 1e-5
    }

    #[test]
    fn rpy_round_trip() {
        let q = Quat::from_rpy(0.3, -0.2, 1.4);
        let rpy = q.to_rpy();
        assert!(close(rpy, Vec3::new(0.3, -0.2, 1.4)));
    }

    #[test]
    fn matrix_matches_rotate() {
        let q = Quat::from_rpy(0.5, 0.1, -0.7);
        let v = Vec3::new(0.2, -1.0, 3.0);
        assert!(close(q.to_mat().mul_vec(v), q.rotate(v)));
        assert!(close(q.inverse_rotate(q.rotate(v)), v));
    }

    #[test]
    fn yaw_rotates_x_to_y() {
        let q = Quat::from_rpy(0.0, 0.0, core::f32::consts::FRAC_PI_2);
        assert!(close(q.rotate(Vec3::X), Vec3::Y));
    }

    #[test]
    fn integrate_body_rate() {
        let mut q = Quat::IDENTITY;
        for _ in 0..1000 {
            q = q.integrate(Vec3::new(0.0, 0.0, 1.0), 0.001);
        }
        assert!((q.to_rpy().z - 1.0).abs() < 1e-4);
    }
}

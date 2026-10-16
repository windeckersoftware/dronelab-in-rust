//! The rotation helpers upstream calls, reproduced formula for formula.
//!
//! Upstream mixes three libraries with different conventions:
//! - PyBullet (`getMatrixFromQuaternion`, `getEulerFromQuaternion`,
//!   `getQuaternionFromEuler`): quaternions `[x, y, z, w]`, Euler angles as
//!   roll/pitch/yaw (extrinsic x-y-z, i.e. Bullet's `getEulerZYX`);
//! - SciPy's `Rotation` with the `'XYZ'` sequence, which is *intrinsic*
//!   x-y'-z'' and so a different set of angles for the same rotation;
//! - transforms3d (used by `CTBRControl`): quaternions `[w, x, y, z]`.
//!
//! Keeping each one separate is what lets the controllers match upstream.

pub type Vec3 = [f64; 3];
/// Row-major 3x3 matrix.
pub type Mat3 = [[f64; 3]; 3];
/// PyBullet / SciPy quaternion order.
pub type QuatXyzw = [f64; 4];
/// transforms3d quaternion order.
pub type QuatWxyz = [f64; 4];

pub fn add(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
pub fn sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
pub fn scale(a: Vec3, s: f64) -> Vec3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
pub fn mul(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] * b[0], a[1] * b[1], a[2] * b[2]]
}
pub fn dot(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
pub fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
/// `np.linalg.norm` of a 3-vector.
pub fn norm(a: Vec3) -> f64 {
    dot(a, a).sqrt()
}
pub fn mat_vec(m: &Mat3, v: Vec3) -> Vec3 {
    [dot(m[0], v), dot(m[1], v), dot(m[2], v)]
}
pub fn transpose(m: &Mat3) -> Mat3 {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}
pub fn mat_mul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut r = [[0.0; 3]; 3];
    for (i, row) in r.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    r
}
pub fn col(m: &Mat3, j: usize) -> Vec3 {
    [m[0][j], m[1][j], m[2][j]]
}
pub fn from_cols(c0: Vec3, c1: Vec3, c2: Vec3) -> Mat3 {
    [
        [c0[0], c1[0], c2[0]],
        [c0[1], c1[1], c2[1]],
        [c0[2], c1[2], c2[2]],
    ]
}

/// `np.clip` for one value (NaN passes through, as in NumPy).
pub fn clip(x: f64, lo: f64, hi: f64) -> f64 {
    if x < lo {
        lo
    } else if x > hi {
        hi
    } else {
        x
    }
}

// ---------------------------------------------------------------- PyBullet

/// `p.getMatrixFromQuaternion` (Bullet's `btMatrix3x3::setRotation`).
pub fn pb_matrix(q: QuatXyzw) -> Mat3 {
    let [x, y, z, w] = q;
    let d = x * x + y * y + z * z + w * w;
    let s = 2.0 / d;
    let (xs, ys, zs) = (x * s, y * s, z * s);
    let (wx, wy, wz) = (w * xs, w * ys, w * zs);
    let (xx, xy, xz) = (x * xs, x * ys, x * zs);
    let (yy, yz, zz) = (y * ys, y * zs, z * zs);
    [
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)],
    ]
}

/// `p.getEulerFromQuaternion`: roll, pitch, yaw (Bullet's `getEulerZYX`).
pub fn pb_euler(q: QuatXyzw) -> Vec3 {
    let [x, y, z, w] = q;
    let (sqx, sqy, sqz, squ) = (x * x, y * y, z * z, w * w);
    let sarg = -2.0 * (x * z - w * y);
    if sarg <= -0.99999 {
        [0.0, -0.5 * std::f64::consts::PI, 2.0 * x.atan2(-y)]
    } else if sarg >= 0.99999 {
        [0.0, 0.5 * std::f64::consts::PI, 2.0 * (-x).atan2(y)]
    } else {
        [
            (2.0 * (y * z + w * x)).atan2(squ - sqx - sqy + sqz),
            sarg.asin(),
            (2.0 * (x * y + w * z)).atan2(squ + sqx - sqy - sqz),
        ]
    }
}

/// `p.getQuaternionFromEuler(roll, pitch, yaw)`.
pub fn pb_quat_from_euler(e: Vec3) -> QuatXyzw {
    let (phi, theta, psi) = (e[0] / 2.0, e[1] / 2.0, e[2] / 2.0);
    let (sp, cp) = phi.sin_cos();
    let (st, ct) = theta.sin_cos();
    let (ss, cs) = psi.sin_cos();
    [
        sp * ct * cs - cp * st * ss,
        cp * st * cs + sp * ct * ss,
        cp * ct * ss - sp * st * cs,
        cp * ct * cs + sp * st * ss,
    ]
}

// ------------------------------------------------------------------- SciPy

/// `Rotation.from_quat(q)` normalisation followed by `as_matrix()`.
pub fn scipy_quat_to_matrix(q: QuatXyzw) -> Mat3 {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    let [x, y, z, w] = [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
    let (x2, y2, z2, w2) = (x * x, y * y, z * z, w * w);
    let (xy, zw, xz, yw, yz, xw) = (x * y, z * w, x * z, y * w, y * z, x * w);
    [
        [x2 - y2 - z2 + w2, 2.0 * (xy - zw), 2.0 * (xz + yw)],
        [2.0 * (xy + zw), -x2 + y2 - z2 + w2, 2.0 * (yz - xw)],
        [2.0 * (xz - yw), 2.0 * (yz + xw), -x2 - y2 + z2 + w2],
    ]
}

/// `Rotation.from_matrix(m)` (Markley's method), scalar-last.
pub fn scipy_matrix_to_quat(m: &Mat3) -> QuatXyzw {
    let decision = [m[0][0], m[1][1], m[2][2], m[0][0] + m[1][1] + m[2][2]];
    let mut choice = 0;
    for k in 1..4 {
        if decision[k] > decision[choice] {
            choice = k;
        }
    }
    let mut q = [0.0; 4];
    if choice != 3 {
        let i = choice;
        let j = (i + 1) % 3;
        let k = (j + 1) % 3;
        q[i] = 1.0 - decision[3] + 2.0 * m[i][i];
        q[j] = m[j][i] + m[i][j];
        q[k] = m[k][i] + m[i][k];
        q[3] = m[k][j] - m[j][k];
    } else {
        q[0] = m[2][1] - m[1][2];
        q[1] = m[0][2] - m[2][0];
        q[2] = m[1][0] - m[0][1];
        q[3] = 1.0 + decision[3];
    }
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
}

/// `Rotation.as_euler('XYZ')` (intrinsic), via SciPy's quaternion algorithm
/// (Bernardes & Viollet, 2022).
pub fn scipy_quat_to_euler_xyz_intrinsic(q: QuatXyzw) -> Vec3 {
    use std::f64::consts::PI;
    // Intrinsic 'XYZ' is extrinsic 'zyx' reversed: i, j, k = z, y, x.
    let (i, j, k) = (2usize, 1usize, 0usize);
    let sign = ((i as i64 - j as i64) * (j as i64 - k as i64) * (k as i64 - i as i64) / 2) as f64;
    let a = q[3] - q[j];
    let b = q[i] + q[k] * sign;
    let c = q[j] + q[3];
    let d = q[k] * sign - q[i];
    let mut angles = [0.0f64; 3];
    angles[1] = 2.0 * c.hypot(d).atan2(a.hypot(b));
    let eps = 1e-7;
    let case = if angles[1].abs() <= eps {
        1
    } else if (angles[1] - PI).abs() <= eps {
        2
    } else {
        0
    };
    let half_sum = b.atan2(a);
    let half_diff = d.atan2(c);
    if case == 0 {
        angles[0] = half_sum - half_diff;
        angles[2] = half_sum + half_diff;
    } else {
        // Gimbal lock: SciPy sets the third angle (of the extrinsic
        // sequence) to zero; for an intrinsic sequence that is the first.
        angles[2] = 0.0;
        angles[0] = if case == 1 {
            2.0 * half_sum
        } else {
            -2.0 * half_diff
        };
    }
    angles[2] *= sign;
    angles[1] -= PI / 2.0;
    angles.swap(0, 2);
    for a in angles.iter_mut() {
        if *a < -PI {
            *a += 2.0 * PI;
        } else if *a > PI {
            *a -= 2.0 * PI;
        }
    }
    angles
}

/// `Rotation.from_matrix(m).as_euler('XYZ')`.
pub fn scipy_matrix_to_euler_xyz(m: &Mat3) -> Vec3 {
    scipy_quat_to_euler_xyz_intrinsic(scipy_matrix_to_quat(m))
}

fn compose_xyzw(p: QuatXyzw, q: QuatXyzw) -> QuatXyzw {
    let c = cross([p[0], p[1], p[2]], [q[0], q[1], q[2]]);
    [
        p[3] * q[0] + q[3] * p[0] + c[0],
        p[3] * q[1] + q[3] * p[1] + c[1],
        p[3] * q[2] + q[3] * p[2] + c[2],
        p[3] * q[3] - p[0] * q[0] - p[1] * q[1] - p[2] * q[2],
    ]
}

fn elementary(axis: usize, angle: f64) -> QuatXyzw {
    let mut q = [0.0, 0.0, 0.0, (angle / 2.0).cos()];
    q[axis] = (angle / 2.0).sin();
    q
}

/// `Rotation.from_euler('XYZ', e).as_quat()` (intrinsic).
pub fn scipy_euler_xyz_to_quat(e: Vec3) -> QuatXyzw {
    let q = compose_xyzw(elementary(0, e[0]), elementary(1, e[1]));
    compose_xyzw(q, elementary(2, e[2]))
}

/// `Rotation.from_euler('XYZ', e).as_matrix()`.
pub fn scipy_euler_xyz_to_matrix(e: Vec3) -> Mat3 {
    scipy_quat_to_matrix(scipy_euler_xyz_to_quat(e))
}

// ------------------------------------------------------------- transforms3d

/// `transforms3d.quaternions.qmult`.
pub fn t3d_qmult(q1: QuatWxyz, q2: QuatWxyz) -> QuatWxyz {
    let [w1, x1, y1, z1] = q1;
    let [w2, x2, y2, z2] = q2;
    [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2,
        w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2,
    ]
}

/// `transforms3d.quaternions.qconjugate`.
pub fn t3d_qconjugate(q: QuatWxyz) -> QuatWxyz {
    [q[0], -q[1], -q[2], -q[3]]
}

/// `transforms3d.quaternions.rotate_vector` (quaternion assumed unit).
pub fn t3d_rotate_vector(v: Vec3, q: QuatWxyz) -> Vec3 {
    let r = t3d_qmult(q, t3d_qmult([0.0, v[0], v[1], v[2]], t3d_qconjugate(q)));
    [r[1], r[2], r[3]]
}

/// `transforms3d.utils.normalized_vector`.
pub fn t3d_normalized(v: Vec3) -> Vec3 {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / n, v[1] / n, v[2] / n]
}

/// `transforms3d.quaternions.mat2quat`, with its positive-w convention.
///
/// transforms3d solves an eigenproblem (Bar-Itzhack); for a rotation matrix
/// that has the same answer as the closed form used here, up to rounding.
pub fn t3d_mat2quat(m: &Mat3) -> QuatWxyz {
    let q = scipy_matrix_to_quat(m);
    let q = [q[3], q[0], q[1], q[2]];
    if q[0] < 0.0 {
        [-q[0], -q[1], -q[2], -q[3]]
    } else {
        q
    }
}

/// Bullet's `btMatrix3x3::getRotation`: matrix to `[x, y, z, w]`.
pub fn bt_matrix_to_quat(m: &Mat3) -> QuatXyzw {
    let trace = m[0][0] + m[1][1] + m[2][2];
    let mut t = [0.0; 4];
    if trace > 0.0 {
        let s = (trace + 1.0).sqrt();
        t[3] = s * 0.5;
        let s = 0.5 / s;
        t[0] = (m[2][1] - m[1][2]) * s;
        t[1] = (m[0][2] - m[2][0]) * s;
        t[2] = (m[1][0] - m[0][1]) * s;
    } else {
        let i = if m[0][0] < m[1][1] {
            if m[1][1] < m[2][2] { 2 } else { 1 }
        } else if m[0][0] < m[2][2] {
            2
        } else {
            0
        };
        let j = (i + 1) % 3;
        let k = (i + 2) % 3;
        let s = (m[i][i] - m[j][j] - m[k][k] + 1.0).sqrt();
        t[i] = s * 0.5;
        let s = 0.5 / s;
        t[3] = (m[k][j] - m[j][k]) * s;
        t[j] = (m[j][i] + m[i][j]) * s;
        t[k] = (m[k][i] + m[i][k]) * s;
    }
    t
}

/// What `p.getBasePositionAndOrientation` returns for an orientation set
/// as `q`: PyBullet goes through a rotation matrix, so the quaternion comes
/// back normalised and with Bullet's choice of sign.
pub fn pb_readback_quat(q: QuatXyzw) -> QuatXyzw {
    bt_matrix_to_quat(&pb_matrix(q))
}

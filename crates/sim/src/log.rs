//! Flight logs that open in ArduPilot and PX4 tools, so simulated flights
//! can be compared with real ones.
//!
//! [`FlightLog`] records one [`Env`] every control step (start it with
//! [`Env::start_log`]) and writes either
//! - an ArduPilot DataFlash `.bin` (MAVExplorer, UAV Log Viewer, pymavlink), or
//! - a PX4 ULog `.ulg` (PX4 Flight Review, PlotJuggler, pyulog).
//!
//! # Frames
//!
//! dronelab's world is z-up and is taken as ENU (x east, y north); its body
//! frame is FLU (x forward, y left, z up). Both log formats are NED / FRD,
//! so positions become north = y, east = x, down = -z, body vectors become
//! (x, -y, -z), and the attitude is rotated to match: a dronelab yaw of 0
//! (nose along +x, east) is a heading of 90 degrees.
//!
//! Latitude and longitude come from a flat-earth projection around
//! [`GeoOrigin`]. Rotor commands (fraction of max thrust, 0..1) are logged
//! as PWM, 1000 + 1000 * command, on output `i` = rotor `i`.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

use dronelab_core::MAX_ROTORS;
use dronelab_core::math::{Quat, Vec3};

use crate::env::Env;

/// Where the local frame's origin is on Earth.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeoOrigin {
    pub lat_deg: f64,
    pub lon_deg: f64,
    /// Altitude above mean sea level, m.
    pub alt_m: f64,
}

impl Default for GeoOrigin {
    /// ArduPilot SITL's default home (Canberra Model Aircraft Club).
    fn default() -> Self {
        GeoOrigin {
            lat_deg: -35.363261,
            lon_deg: 149.165230,
            alt_m: 584.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogFormat {
    /// ArduPilot DataFlash binary, `.bin`.
    DataFlash,
    /// PX4 ULog, `.ulg`.
    ULog,
}

impl LogFormat {
    /// `"bin"`, `"dataflash"`, `"ulg"`, `"ulog"` (case-insensitive).
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "bin" | "dataflash" | "ardupilot" => Some(LogFormat::DataFlash),
            "ulg" | "ulog" | "px4" => Some(LogFormat::ULog),
            _ => None,
        }
    }

    /// From a path's extension.
    pub fn from_path(path: &Path) -> Option<Self> {
        path.extension()?.to_str().and_then(Self::parse)
    }
}

/// One control step of one vehicle, in dronelab's own frames.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sample {
    /// Log time, s; continues across episodes.
    pub time: f64,
    /// Episode number within this log, from 0.
    pub episode: u32,
    pub pos: Vec3,
    pub vel: Vec3,
    pub quat: Quat,
    /// IMU body rates and specific force, FLU.
    pub gyro: Vec3,
    pub accel: Vec3,
    /// Commanded fraction of max thrust per rotor.
    pub cmd: [f32; MAX_ROTORS],
    /// Rotor speeds, rad/s.
    pub omega: [f32; MAX_ROTORS],
    pub voltage: f32,
    pub current: f32,
    pub soc: f32,
    /// Charge and energy drawn since the log started, mAh and Wh.
    pub used_mah: f32,
    pub used_wh: f32,
}

/// A recorded flight; see the module docs.
#[derive(Debug, Clone)]
pub struct FlightLog {
    pub samples: Vec<Sample>,
    /// Episode starts: (log time, seed).
    pub episodes: Vec<(f64, u64)>,
    pub origin: GeoOrigin,
    pub vehicle: &'static str,
    pub rotors: usize,
    pub cells: u32,
    pub capacity_ah: f32,
    pub r_internal: f32,
    control_hz: u32,
    time_base: f64,
}

/// GPS time of the log start: 2026-01-01 00:00:00 UTC (week, ms of week).
const GPS_START: (u16, u32) = (2399, 345_618_000);

impl FlightLog {
    pub fn new(env: &Env, origin: GeoOrigin) -> Self {
        let cfg = env.config();
        FlightLog {
            samples: Vec::new(),
            episodes: Vec::new(),
            origin,
            vehicle: cfg.airframe.vehicle.name,
            rotors: cfg.airframe.vehicle.rotor_count(),
            cells: cfg.airframe.battery.cells,
            capacity_ah: cfg.airframe.battery.capacity_ah,
            r_internal: cfg.airframe.battery.r_internal,
            control_hz: cfg.control_hz,
            time_base: 0.0,
        }
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Adds the env's current state. `new_episode` marks the first state of
    /// an episode; log time then continues one control period after the
    /// previous sample.
    pub fn record(&mut self, env: &Env, new_episode: bool, cmd: &[f32; MAX_ROTORS]) {
        let d = &env.drone;
        let prev = self.samples.last().copied();
        if new_episode && let Some(p) = prev {
            self.time_base = p.time + 1.0 / self.control_hz as f64;
        }
        let time = self.time_base + env.time();
        if new_episode || prev.is_none() {
            self.episodes.push((time, env.seed));
        }
        let (mut used_mah, mut used_wh) = (0.0, 0.0);
        if let Some(p) = prev {
            let dt = (time - p.time).max(0.0) as f32;
            used_mah = p.used_mah + d.battery.current * dt / 3.6;
            used_wh = p.used_wh + d.battery.current * d.battery.voltage * dt / 3600.0;
        }
        self.samples.push(Sample {
            time,
            episode: self.episodes.len() as u32 - 1,
            pos: d.state.pos,
            vel: d.state.vel,
            quat: d.state.quat,
            gyro: d.imu.last.gyro,
            accel: d.imu.last.accel,
            cmd: *cmd,
            omega: d.omega,
            voltage: d.battery.voltage,
            current: d.battery.current,
            soc: d.battery.soc,
            used_mah,
            used_wh,
        });
    }

    pub fn save(&self, path: impl AsRef<Path>, format: LogFormat) -> io::Result<()> {
        let mut w = BufWriter::new(File::create(path)?);
        match format {
            LogFormat::DataFlash => self.write_dataflash(&mut w)?,
            LogFormat::ULog => self.write_ulog(&mut w)?,
        }
        w.flush()
    }

    /// Writes an ArduPilot DataFlash log: FMT records, then PARM, MSG,
    /// ATT, IMU, GPS, POS, BAT, RCOU (+ RCO2/RCO3 beyond 14 rotors), ESC and
    /// XKF1 every sample.
    pub fn write_dataflash<W: Write>(&self, w: &mut W) -> io::Result<()> {
        let mut df = DataFlash { w };
        for f in DF_FORMATS {
            df.fmt(f)?;
        }
        let t0 = self.samples.first().map_or(0, |s| us(s.time));
        let params = [
            ("BATT_CAPACITY", self.capacity_ah * 1000.0),
            ("BATT_MONITOR", 4.0),
            ("FRAME_CLASS", frame_class(self.rotors)),
        ];
        for (name, value) in params {
            df.msg(DF_PARM)?.u64(t0)?.str(name, 16)?.f32(value)?;
        }
        df.msg(DF_MSG)?.u64(t0)?.str(
            &format!("dronelab {} {}", env!("CARGO_PKG_VERSION"), self.vehicle),
            64,
        )?;
        df.msg(DF_MSG)?
            .u64(t0)?
            .str(&format!("Frame: {} rotors", self.rotors), 64)?;

        let mut episodes = self.episodes.iter().peekable();
        for s in &self.samples {
            let t = us(s.time);
            if let Some((_, seed)) = episodes.next_if(|(et, _)| us(*et) <= t) {
                df.msg(DF_MSG)?
                    .u64(t)?
                    .str(&format!("episode {} seed {seed}", s.episode), 64)?;
            }
            let n = Ned::from(s, &self.origin);
            let (roll, pitch, yaw) = n.euler_deg();

            df.msg(DF_ATT)?
                .u64(t)?
                .i16(cdeg(roll))?
                .i16(cdeg(pitch))?
                .u16(cdeg(yaw) as u16)?;

            df.msg(DF_IMU)?.u64(t)?.u8(0)?;
            for v in n.gyro.iter().chain(&n.accel) {
                df.f32(*v as f32)?;
            }
            // EG, EA, T, GH, AH, GHz, AHz
            let hz = self.control_hz as u16;
            df.u32(0)?
                .u32(0)?
                .f32(25.0)?
                .u8(1)?
                .u8(1)?
                .u16(hz)?
                .u16(hz)?;

            let (gwk, gms) = gps_time(s.time);
            let speed = n.vel[0].hypot(n.vel[1]);
            let course = n.vel[1].atan2(n.vel[0]).to_degrees().rem_euclid(360.0);
            df.msg(DF_GPS)?
                .u64(t)?
                .u8(0)?
                .u8(3)? // 3D fix
                .u32(gms)?
                .u16(gwk)?
                .u8(12)?
                .i16(80)? // HDop 0.8
                .i32(e7(n.lat))?
                .i32(e7(n.lon))?
                .i32((n.alt * 100.0).round() as i32)?
                .f32(speed as f32)?
                .f32(course as f32)?
                .f32(n.vel[2] as f32)?
                .f32(0.0)?
                .u8(1)?;

            let rel = n.alt - self.origin.alt_m;
            df.msg(DF_POS)?
                .u64(t)?
                .i32(e7(n.lat))?
                .i32(e7(n.lon))?
                .f32(n.alt as f32)?
                .f32(rel as f32)?
                .f32(rel as f32)?;

            let resting = s.voltage + s.current * self.r_internal;
            df.msg(DF_BAT)?
                .u64(t)?
                .u8(0)?
                .f32(s.voltage)?
                .f32(resting)?
                .f32(s.current)?
                .f32(s.used_mah)?
                .f32(s.used_wh)?
                .i16(2500)?
                .f32(self.r_internal)?
                .u8((s.soc * 100.0).round().clamp(0.0, 100.0) as u8)?;

            for (id, range) in [(DF_RCOU, 0..14), (DF_RCO2, 14..18), (DF_RCO3, 18..32)] {
                if range.start >= self.rotors {
                    break;
                }
                df.msg(id)?.u64(t)?;
                for i in range {
                    df.u16(if i < self.rotors { pwm(s.cmd[i]) } else { 0 })?;
                }
            }

            for i in 0..self.rotors {
                let rpm = s.omega[i] * 60.0 / core::f32::consts::TAU;
                df.msg(DF_ESC)?.u64(t)?.u8(i as u8)?.f32(rpm)?;
            }

            df.msg(DF_XKF1)?
                .u64(t)?
                .u8(0)?
                .i16(cdeg(roll))?
                .i16(cdeg(pitch))?
                .u16(cdeg(yaw) as u16)?;
            for v in n.vel {
                df.f32(v as f32)?;
            }
            df.f32(n.vel[2] as f32)?; // dPD
            for v in n.pos {
                df.f32(v as f32)?;
            }
            df.i16(0)?.i16(0)?.i16(0)?.f32(self.origin.alt_m as f32)?;
        }
        Ok(())
    }

    /// Writes a PX4 ULog with vehicle_attitude, vehicle_angular_velocity,
    /// sensor_combined, vehicle_local_position, vehicle_global_position,
    /// battery_status and actuator_outputs (a second instance beyond 16
    /// rotors).
    pub fn write_ulog<W: Write>(&self, w: &mut W) -> io::Result<()> {
        // Header: magic, version 1, start timestamp.
        w.write_all(b"ULog\x01\x12\x35\x01")?;
        let t0 = self.samples.first().map_or(0, |s| us(s.time));
        w.write_all(&t0.to_le_bytes())?;
        let mut u = ULog { w };

        // Flag bits: no compat/incompat flags, no appended data.
        u.message(b'B', &[0u8; 40])?;
        for (name, fields) in ULOG_FORMATS {
            u.message(b'F', format!("{name}:{fields}").as_bytes())?;
        }
        u.info("char[{}] sys_name", "dronelab")?;
        u.info("char[{}] ver_hw", self.vehicle)?;
        u.info("char[{}] ver_sw", env!("CARGO_PKG_VERSION"))?;
        let mav_type: i32 = match self.rotors {
            6 => 13,
            8 => 14,
            _ => 2,
        };
        u.param_i32("MAV_TYPE", mav_type)?;
        u.param_i32("BAT1_N_CELLS", self.cells as i32)?;
        u.param_f32("BAT1_CAPACITY", self.capacity_ah * 1000.0)?;
        u.param_f32("BAT1_R_INTERNAL", self.r_internal)?;

        let outputs = self.rotors.div_ceil(16);
        for (msg_id, (name, _)) in ULOG_FORMATS.iter().enumerate() {
            let instances = if *name == "actuator_outputs" {
                outputs
            } else {
                1
            };
            for multi in 0..instances {
                let id = (msg_id + multi) as u16 + if multi > 0 { ULOG_EXTRA_ID } else { 0 };
                let mut body = vec![multi as u8];
                body.extend_from_slice(&id.to_le_bytes());
                body.extend_from_slice(name.as_bytes());
                u.message(b'A', &body)?;
            }
        }

        let mut episodes = self.episodes.iter().peekable();
        let mut b = Vec::with_capacity(256);
        for s in &self.samples {
            let t = us(s.time);
            if let Some((_, seed)) = episodes.next_if(|(et, _)| us(*et) <= t) {
                b.clear();
                b.push(b'6'); // info
                b.extend_from_slice(&t.to_le_bytes());
                b.extend_from_slice(format!("episode {} seed {seed}", s.episode).as_bytes());
                u.message(b'L', &b)?;
            }
            let n = Ned::from(s, &self.origin);

            // vehicle_attitude
            let mut d = u.data(UL_ATT, t);
            d.u64(t);
            for q in n.quat {
                d.f32(q as f32);
            }
            for q in [1.0, 0.0, 0.0, 0.0] {
                d.f32(q);
            }
            d.u8(0);
            d.write()?;

            let mut d = u.data(UL_RATES, t);
            d.u64(t);
            for v in n.gyro {
                d.f32(v as f32);
            }
            d.write()?;

            let mut d = u.data(UL_SENSORS, t);
            for v in n.gyro {
                d.f32(v as f32);
            }
            let dt_us = 1_000_000 / self.control_hz;
            d.u32(dt_us).i32(0);
            for v in n.accel {
                d.f32(v as f32);
            }
            d.u32(dt_us).u8(0);
            d.write()?;

            let mut d = u.data(UL_LOCAL, t);
            d.u64(t)
                .u64(t0)
                .f64(self.origin.lat_deg)
                .f64(self.origin.lon_deg);
            for v in n.pos.iter().chain(&n.vel) {
                d.f32(*v as f32);
            }
            d.f32(n.vel[2] as f32); // z_deriv
            for v in n.acc {
                d.f32(v as f32);
            }
            d.f32(n.heading_rad() as f32).f32(self.origin.alt_m as f32);
            for _ in 0..6 {
                d.u8(1); // xy/z/v_xy/v_z valid, xy/z global
            }
            d.write()?;

            let mut d = u.data(UL_GLOBAL, t);
            d.u64(t)
                .f64(n.lat)
                .f64(n.lon)
                .f32(n.alt as f32)
                .f32(0.5)
                .f32(0.8);
            d.write()?;

            let mut d = u.data(UL_BATTERY, t);
            d.f32(s.voltage)
                .f32(s.current)
                .f32(s.current)
                .f32(s.used_mah)
                .f32(s.soc)
                .f32(25.0)
                .u8(self.cells.min(255) as u8)
                .u8(1)
                .u8(0);
            d.write()?;

            for k in 0..outputs {
                let id = if k == 0 {
                    UL_OUTPUTS
                } else {
                    UL_OUTPUTS + k as u16 + ULOG_EXTRA_ID
                };
                let mut d = u.data(id, t);
                let lo = 16 * k;
                d.u32((self.rotors - lo).min(16) as u32);
                for i in lo..lo + 16 {
                    d.f32(if i < self.rotors {
                        pwm(s.cmd[i]) as f32
                    } else {
                        0.0
                    });
                }
                d.write()?;
            }
        }
        Ok(())
    }
}

fn us(t: f64) -> u64 {
    (t * 1e6).round().max(0.0) as u64
}

fn cdeg(deg: f64) -> i16 {
    (deg * 100.0).round() as i16
}

fn e7(deg: f64) -> i32 {
    (deg * 1e7).round() as i32
}

fn pwm(cmd: f32) -> u16 {
    (1000.0 + 1000.0 * cmd.clamp(0.0, 1.0)).round() as u16
}

/// ArduPilot FRAME_CLASS by rotor count (1 quad, 2 hexa, 3 octa, 4 octaquad).
fn frame_class(rotors: usize) -> f32 {
    match rotors {
        6 => 2.0,
        8 => 3.0,
        16 => 4.0,
        _ => 1.0,
    }
}

fn gps_time(t: f64) -> (u16, u32) {
    const WEEK_MS: u64 = 604_800_000;
    let ms = GPS_START.1 as u64 + (t * 1000.0).round() as u64;
    (
        (GPS_START.0 as u64 + ms / WEEK_MS) as u16,
        (ms % WEEK_MS) as u32,
    )
}

/// A sample in NED / FRD.
struct Ned {
    pos: [f64; 3],
    vel: [f64; 3],
    /// Acceleration, world NED, from the specific force.
    acc: [f64; 3],
    /// Body FRD to NED, scalar first.
    quat: [f64; 4],
    gyro: [f64; 3],
    accel: [f64; 3],
    lat: f64,
    lon: f64,
    /// Above mean sea level, m.
    alt: f64,
}

const EARTH_RADIUS: f64 = 6_378_137.0;

impl Ned {
    fn from(s: &Sample, o: &GeoOrigin) -> Self {
        let v = |v: Vec3| [v.x as f64, v.y as f64, v.z as f64];
        let world = |v: Vec3| [v.y as f64, v.x as f64, -v.z as f64];
        let body = |v: Vec3| [v.x as f64, -v.y as f64, -v.z as f64];
        let q = [s.quat.w, s.quat.x, s.quat.y, s.quat.z].map(|c| c as f64);
        // q_ned_frd = q_ned_enu * q_enu_flu * q_flu_frd
        let h = core::f64::consts::FRAC_1_SQRT_2;
        let quat = qmul(qmul([0.0, h, h, 0.0], q), [0.0, 1.0, 0.0, 0.0]);
        // World acceleration = R * specific force - gravity (z-up).
        let mut a = v(s.quat.rotate(s.accel));
        a[2] -= dronelab_core::G as f64;
        let acc = [a[1], a[0], -a[2]];
        let pos = world(s.pos);
        let lat = o.lat_deg + (pos[0] / EARTH_RADIUS).to_degrees();
        let lon = o.lon_deg + (pos[1] / (EARTH_RADIUS * o.lat_deg.to_radians().cos())).to_degrees();
        Ned {
            pos,
            vel: world(s.vel),
            acc,
            quat,
            gyro: body(s.gyro),
            accel: body(s.accel),
            lat,
            lon,
            alt: o.alt_m - pos[2],
        }
    }

    /// Roll, pitch, heading (0..360), degrees.
    fn euler_deg(&self) -> (f64, f64, f64) {
        let [w, x, y, z] = self.quat;
        let roll = (2.0 * (w * x + y * z)).atan2(1.0 - 2.0 * (x * x + y * y));
        let pitch = (2.0 * (w * y - z * x)).clamp(-1.0, 1.0).asin();
        (
            roll.to_degrees(),
            pitch.to_degrees(),
            self.heading_rad().to_degrees().rem_euclid(360.0),
        )
    }

    fn heading_rad(&self) -> f64 {
        let [w, x, y, z] = self.quat;
        (2.0 * (w * z + x * y)).atan2(1.0 - 2.0 * (y * y + z * z))
    }
}

fn qmul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}

// ---------------------------------------------------------------------------
// DataFlash

struct DfFormat {
    id: u8,
    name: &'static str,
    format: &'static str,
    columns: &'static str,
}

const DF_FMT: u8 = 128;
const DF_PARM: u8 = 129;
const DF_MSG: u8 = 130;
const DF_ATT: u8 = 131;
const DF_IMU: u8 = 132;
const DF_GPS: u8 = 133;
const DF_POS: u8 = 134;
const DF_BAT: u8 = 135;
const DF_RCOU: u8 = 136;
const DF_RCO2: u8 = 137;
const DF_RCO3: u8 = 138;
const DF_ESC: u8 = 139;
const DF_XKF1: u8 = 140;

const DF_FORMATS: &[DfFormat] = &[
    DfFormat {
        id: DF_FMT,
        name: "FMT",
        format: "BBnNZ",
        columns: "Type,Length,Name,Format,Columns",
    },
    DfFormat {
        id: DF_PARM,
        name: "PARM",
        format: "QNf",
        columns: "TimeUS,Name,Value",
    },
    DfFormat {
        id: DF_MSG,
        name: "MSG",
        format: "QZ",
        columns: "TimeUS,Message",
    },
    DfFormat {
        id: DF_ATT,
        name: "ATT",
        format: "QccC",
        columns: "TimeUS,Roll,Pitch,Yaw",
    },
    DfFormat {
        id: DF_IMU,
        name: "IMU",
        format: "QBffffffIIfBBHH",
        columns: "TimeUS,I,GyrX,GyrY,GyrZ,AccX,AccY,AccZ,EG,EA,T,GH,AH,GHz,AHz",
    },
    DfFormat {
        id: DF_GPS,
        name: "GPS",
        format: "QBBIHBcLLeffffB",
        columns: "TimeUS,I,Status,GMS,GWk,NSats,HDop,Lat,Lng,Alt,Spd,GCrs,VZ,Yaw,U",
    },
    DfFormat {
        id: DF_POS,
        name: "POS",
        format: "QLLfff",
        columns: "TimeUS,Lat,Lng,Alt,RelHomeAlt,RelOriginAlt",
    },
    DfFormat {
        id: DF_BAT,
        name: "BAT",
        format: "QBfffffcfB",
        columns: "TimeUS,Inst,Volt,VoltR,Curr,CurrTot,EnrgTot,Temp,Res,RemPct",
    },
    DfFormat {
        id: DF_RCOU,
        name: "RCOU",
        format: "QHHHHHHHHHHHHHH",
        columns: "TimeUS,C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12,C13,C14",
    },
    DfFormat {
        id: DF_RCO2,
        name: "RCO2",
        format: "QHHHH",
        columns: "TimeUS,C15,C16,C17,C18",
    },
    DfFormat {
        id: DF_RCO3,
        name: "RCO3",
        format: "QHHHHHHHHHHHHHH",
        columns: "TimeUS,C19,C20,C21,C22,C23,C24,C25,C26,C27,C28,C29,C30,C31,C32",
    },
    DfFormat {
        id: DF_ESC,
        name: "ESC",
        format: "QBf",
        columns: "TimeUS,Instance,RPM",
    },
    DfFormat {
        id: DF_XKF1,
        name: "XKF1",
        format: "QBccCfffffffcccf",
        columns: "TimeUS,C,Roll,Pitch,Yaw,VN,VE,VD,dPD,PN,PE,PD,GX,GY,GZ,OH",
    },
];

/// Payload bytes of a DataFlash format string.
fn df_size(format: &str) -> usize {
    format
        .chars()
        .map(|c| match c {
            'b' | 'B' | 'M' => 1,
            'h' | 'H' | 'c' | 'C' => 2,
            'i' | 'I' | 'f' | 'e' | 'E' | 'L' | 'n' => 4,
            'd' | 'q' | 'Q' => 8,
            'N' => 16,
            'Z' | 'a' => 64,
            _ => panic!("unsupported DataFlash format char {c:?}"),
        })
        .sum()
}

struct DataFlash<'a, W: Write> {
    w: &'a mut W,
}

impl<W: Write> DataFlash<'_, W> {
    fn msg(&mut self, id: u8) -> io::Result<&mut Self> {
        self.w.write_all(&[0xA3, 0x95, id])?;
        Ok(self)
    }

    fn fmt(&mut self, f: &DfFormat) -> io::Result<()> {
        let len = 3 + df_size(f.format);
        self.msg(DF_FMT)?
            .u8(f.id)?
            .u8(len as u8)?
            .str(f.name, 4)?
            .str(f.format, 16)?
            .str(f.columns, 64)?;
        Ok(())
    }

    /// Zero-padded (and truncated) to `n` bytes.
    fn str(&mut self, s: &str, n: usize) -> io::Result<&mut Self> {
        let mut buf = [0u8; 64];
        let k = s.len().min(n);
        buf[..k].copy_from_slice(&s.as_bytes()[..k]);
        self.w.write_all(&buf[..n])?;
        Ok(self)
    }
}

macro_rules! df_put {
    ($($name:ident: $t:ty),*) => {
        impl<W: Write> DataFlash<'_, W> {
            $(fn $name(&mut self, v: $t) -> io::Result<&mut Self> {
                self.w.write_all(&v.to_le_bytes())?;
                Ok(self)
            })*
        }
    };
}
df_put!(u8: u8, u16: u16, i16: i16, u32: u32, i32: i32, u64: u64, f32: f32);

// ---------------------------------------------------------------------------
// ULog

/// Topic formats, a subset of PX4's fields in PX4's order (no padding
/// fields: readers parse the declared layout).
const ULOG_FORMATS: &[(&str, &str)] = &[
    (
        "vehicle_attitude",
        "uint64_t timestamp;uint64_t timestamp_sample;float[4] q;float[4] delta_q_reset;\
         uint8_t quat_reset_counter;",
    ),
    (
        "vehicle_angular_velocity",
        "uint64_t timestamp;uint64_t timestamp_sample;float[3] xyz;",
    ),
    (
        "sensor_combined",
        "uint64_t timestamp;float[3] gyro_rad;uint32_t gyro_integral_dt;\
         int32_t accelerometer_timestamp_relative;float[3] accelerometer_m_s2;\
         uint32_t accelerometer_integral_dt;uint8_t accelerometer_clipping;",
    ),
    (
        "vehicle_local_position",
        "uint64_t timestamp;uint64_t timestamp_sample;uint64_t ref_timestamp;double ref_lat;\
         double ref_lon;float x;float y;float z;float vx;float vy;float vz;float z_deriv;\
         float ax;float ay;float az;float heading;float ref_alt;bool xy_valid;bool z_valid;\
         bool v_xy_valid;bool v_z_valid;bool xy_global;bool z_global;",
    ),
    (
        "vehicle_global_position",
        "uint64_t timestamp;uint64_t timestamp_sample;double lat;double lon;float alt;\
         float eph;float epv;",
    ),
    (
        "battery_status",
        "uint64_t timestamp;float voltage_v;float current_a;float current_average_a;\
         float discharged_mah;float remaining;float temperature;uint8_t cell_count;\
         bool connected;uint8_t id;",
    ),
    (
        "actuator_outputs",
        "uint64_t timestamp;uint32_t noutputs;float[16] output;",
    ),
];

// Message ids: the index in ULOG_FORMATS.
const UL_ATT: u16 = 0;
const UL_RATES: u16 = 1;
const UL_SENSORS: u16 = 2;
const UL_LOCAL: u16 = 3;
const UL_GLOBAL: u16 = 4;
const UL_BATTERY: u16 = 5;
const UL_OUTPUTS: u16 = 6;
/// Offset for the ids of second and later instances of a topic.
const ULOG_EXTRA_ID: u16 = 100;

struct ULog<W: Write> {
    w: W,
}

impl<W: Write> ULog<W> {
    fn message(&mut self, kind: u8, body: &[u8]) -> io::Result<()> {
        let len = u16::try_from(body.len())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "ULog message too long"))?;
        self.w.write_all(&len.to_le_bytes())?;
        self.w.write_all(&[kind])?;
        self.w.write_all(body)
    }

    /// Info message; `{}` in `key` becomes the value's length.
    fn info(&mut self, key: &str, value: &str) -> io::Result<()> {
        let key = key.replace("{}", &value.len().to_string());
        let mut body = vec![key.len() as u8];
        body.extend_from_slice(key.as_bytes());
        body.extend_from_slice(value.as_bytes());
        self.message(b'I', &body)
    }

    fn param(&mut self, key: &str, value: [u8; 4]) -> io::Result<()> {
        let mut body = vec![key.len() as u8];
        body.extend_from_slice(key.as_bytes());
        body.extend_from_slice(&value);
        self.message(b'P', &body)
    }

    fn param_i32(&mut self, name: &str, v: i32) -> io::Result<()> {
        self.param(&format!("int32_t {name}"), v.to_le_bytes())
    }

    fn param_f32(&mut self, name: &str, v: f32) -> io::Result<()> {
        self.param(&format!("float {name}"), v.to_le_bytes())
    }

    /// A data message for subscription `id`, starting with its timestamp.
    fn data(&mut self, id: u16, t: u64) -> ULogData<'_, W> {
        let mut buf = Vec::with_capacity(128);
        buf.extend_from_slice(&id.to_le_bytes());
        buf.extend_from_slice(&t.to_le_bytes());
        ULogData { log: self, buf }
    }
}

struct ULogData<'a, W: Write> {
    log: &'a mut ULog<W>,
    buf: Vec<u8>,
}

impl<W: Write> ULogData<'_, W> {
    fn write(self) -> io::Result<()> {
        self.log.message(b'D', &self.buf)
    }
}

macro_rules! ul_put {
    ($($name:ident: $t:ty),*) => {
        impl<W: Write> ULogData<'_, W> {
            $(fn $name(&mut self, v: $t) -> &mut Self {
                self.buf.extend_from_slice(&v.to_le_bytes());
                self
            })*
        }
    };
}
ul_put!(u8: u8, u32: u32, i32: i32, u64: u64, f32: f32, f64: f64);

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(quat: Quat) -> Sample {
        Sample {
            time: 0.0,
            episode: 0,
            pos: Vec3::new(3.0, 4.0, 10.0),
            vel: Vec3::new(1.0, 2.0, 0.5),
            quat,
            gyro: Vec3::new(0.1, 0.2, 0.3),
            accel: Vec3::new(0.0, 0.0, 9.81),
            cmd: [0.0; MAX_ROTORS],
            omega: [0.0; MAX_ROTORS],
            voltage: 0.0,
            current: 0.0,
            soc: 1.0,
            used_mah: 0.0,
            used_wh: 0.0,
        }
    }

    #[test]
    fn enu_flu_to_ned_frd() {
        let o = GeoOrigin::default();
        let n = Ned::from(&sample(Quat::IDENTITY), &o);
        assert_eq!(n.pos, [4.0, 3.0, -10.0]);
        assert_eq!(n.vel, [2.0, 1.0, -0.5]);
        assert!((n.gyro[0] - 0.1).abs() < 1e-6 && (n.gyro[1] + 0.2).abs() < 1e-6);
        assert!(
            (n.accel[2] + 9.81).abs() < 1e-5,
            "level IMU reads -g in FRD"
        );
        assert!((n.alt - 594.0).abs() < 1e-9);
        // Nose along +x (east) is a heading of 90 degrees, level.
        let (r, p, y) = n.euler_deg();
        assert!(
            r.abs() < 1e-9 && p.abs() < 1e-9 && (y - 90.0).abs() < 1e-9,
            "{r} {p} {y}"
        );

        // Roll is about the shared forward axis and keeps its sign; +pitch
        // about +y (left) is nose down, so it flips; +yaw is counter-
        // clockwise from east, so the heading is 90 degrees minus it.
        let q = Quat::from_rpy(0.2, 0.1, 0.5);
        let (r, p, y) = Ned::from(&sample(q), &o).euler_deg();
        assert!((r - 0.2f64.to_degrees()).abs() < 1e-3, "{r}");
        assert!((p + 0.1f64.to_degrees()).abs() < 1e-3, "{p}");
        assert!((y - (90.0 - 0.5f64.to_degrees())).abs() < 1e-3, "{y}");
    }

    #[test]
    fn gps_week_rolls_over() {
        assert_eq!(gps_time(0.0), GPS_START);
        let (wk, ms) = gps_time(604_800.0);
        assert_eq!((wk, ms), (GPS_START.0 + 1, GPS_START.1));
    }

    #[test]
    fn format_lengths_fit() {
        for f in DF_FORMATS {
            assert!(3 + df_size(f.format) <= 255, "{}", f.name);
            assert_eq!(f.format.len(), f.columns.split(',').count(), "{}", f.name);
            assert!(f.columns.len() <= 64 && f.format.len() <= 16);
        }
        assert_eq!(3 + df_size("BBnNZ"), 89);
    }
}

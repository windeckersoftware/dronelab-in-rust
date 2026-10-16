//! Racing: gates, tracks, lap timing and a persistent leaderboard.

use std::path::Path;

use dronelab_core::math::{Vec3, cos, sin};
use serde::{Deserialize, Serialize};

use crate::airframe::Airframe;

/// A rectangular gate, flown through along its normal.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Gate {
    pub center: [f32; 3],
    /// Heading of the direction of travel through the gate, rad.
    pub yaw: f32,
    /// Opening width and height, m.
    pub width: f32,
    pub height: f32,
    /// Frame thickness around the opening, m. Touching it is a crash.
    pub frame: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Crossing {
    /// The gate plane was not crossed forwards this step.
    None,
    /// Through the opening with the whole vehicle inside it.
    Through,
    /// Hit the frame.
    Frame,
    /// Crossed the plane outside the gate.
    Missed,
}

impl Gate {
    pub fn center(&self) -> Vec3 {
        Vec3::from_array(self.center)
    }

    /// Unit direction of travel.
    pub fn normal(&self) -> Vec3 {
        Vec3::new(cos(self.yaw), sin(self.yaw), 0.0)
    }

    /// Classifies the move `prev -> cur` of a vehicle of `radius` metres.
    pub fn crossing(&self, prev: Vec3, cur: Vec3, radius: f32) -> Crossing {
        let (c, n) = (self.center(), self.normal());
        let (a, b) = ((prev - c).dot(n), (cur - c).dot(n));
        if !(a < 0.0 && b >= 0.0) {
            return Crossing::None;
        }
        let hit = prev + (cur - prev) * (-a / (b - a));
        let lateral = Vec3::new(-n.y, n.x, 0.0);
        let (u, v) = (((hit - c).dot(lateral)).abs(), (hit.z - c.z).abs());
        let (hw, hh) = (0.5 * self.width, 0.5 * self.height);
        if u + radius <= hw && v + radius <= hh {
            Crossing::Through
        } else if u - radius <= hw + self.frame && v - radius <= hh + self.frame {
            Crossing::Frame
        } else {
            Crossing::Missed
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Track {
    pub name: String,
    pub gates: Vec<Gate>,
}

impl Track {
    /// `n` gates evenly spaced on a counter-clockwise circle.
    pub fn circle(
        name: &str,
        n: usize,
        radius: f32,
        height: f32,
        gate_w: f32,
        gate_h: f32,
    ) -> Self {
        let gates = (0..n)
            .map(|k| {
                let a = k as f32 * 2.0 * core::f32::consts::PI / n as f32;
                Gate {
                    center: [radius * cos(a), radius * sin(a), height],
                    yaw: a + core::f32::consts::FRAC_PI_2,
                    width: gate_w,
                    height: gate_h,
                    frame: 0.1 * gate_w,
                }
            })
            .collect();
        Track {
            name: name.to_string(),
            gates,
        }
    }

    /// The default circuit for an airframe, scaled to its size.
    pub fn for_airframe(af: &Airframe) -> Self {
        let r = af.radius();
        let height = (5.0 * r).max(1.0);
        Track::circle(
            &format!("circle_{}", af.vehicle.name),
            6,
            25.0 * r,
            height,
            10.0 * r,
            8.0 * r,
        )
    }

    pub fn load(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        serde_json::from_str(&text).map_err(std::io::Error::other)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        std::fs::write(
            path,
            serde_json::to_string_pretty(self).map_err(std::io::Error::other)?,
        )
    }
}

/// Tracks progress through a track and times laps.
///
/// Gate 0 is start/finish: the clock starts there and a lap ends at the next
/// pass of gate 0 after all other gates in order.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LapTimer {
    /// Index of the gate to fly through next.
    pub next_gate: usize,
    pub gates_passed: u32,
    lap_start: Option<f64>,
    pub laps: Vec<f64>,
}

impl LapTimer {
    /// Records passing the expected gate at time `t` (s). Returns the lap
    /// time if this pass completed a lap.
    pub fn pass(&mut self, gate_count: usize, t: f64) -> Option<f64> {
        let mut finished = None;
        if self.next_gate == 0 {
            finished = self.lap_start.map(|start| t - start);
            self.laps.extend(finished);
            self.lap_start = Some(t);
        }
        self.gates_passed += 1;
        self.next_gate = (self.next_gate + 1) % gate_count;
        finished
    }

    pub fn best_lap(&self) -> Option<f64> {
        self.laps.iter().copied().reduce(f64::min)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LapRecord {
    pub pilot: String,
    pub vehicle: String,
    pub track: String,
    pub lap_time: f64,
    /// Episode seed, so the lap can be replayed exactly.
    pub seed: u64,
}

/// Fastest laps, kept sorted, stored as JSON.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Leaderboard {
    pub entries: Vec<LapRecord>,
    pub capacity: usize,
}

impl Leaderboard {
    pub fn new(capacity: usize) -> Self {
        Leaderboard {
            entries: Vec::new(),
            capacity,
        }
    }

    /// Adds a lap; returns its 1-based rank if it made the board.
    pub fn submit(&mut self, record: LapRecord) -> Option<usize> {
        let pos = self
            .entries
            .partition_point(|e| e.lap_time <= record.lap_time);
        if pos >= self.capacity {
            return None;
        }
        self.entries.insert(pos, record);
        self.entries.truncate(self.capacity);
        Some(pos + 1)
    }

    /// Fastest laps on one track and vehicle.
    pub fn top<'a>(
        &'a self,
        track: &'a str,
        vehicle: &'a str,
    ) -> impl Iterator<Item = &'a LapRecord> + 'a {
        self.entries
            .iter()
            .filter(move |e| e.track == track && e.vehicle == vehicle)
    }

    /// Loads a board, or returns an empty one if the file does not exist.
    pub fn load_or_new(path: impl AsRef<Path>, capacity: usize) -> std::io::Result<Self> {
        match std::fs::read_to_string(path) {
            Ok(text) => serde_json::from_str(&text).map_err(std::io::Error::other),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Leaderboard::new(capacity)),
            Err(e) => Err(e),
        }
    }

    pub fn save(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        std::fs::write(
            path,
            serde_json::to_string_pretty(self).map_err(std::io::Error::other)?,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gate() -> Gate {
        Gate {
            center: [0.0, 0.0, 2.0],
            yaw: 0.0,
            width: 2.0,
            height: 1.0,
            frame: 0.2,
        }
    }

    #[test]
    fn classifies_crossings() {
        let g = gate();
        let p = |y: f32, z: f32| (Vec3::new(-0.1, y, z), Vec3::new(0.1, y, z));
        let (a, b) = p(0.0, 2.0);
        assert_eq!(g.crossing(a, b, 0.2), Crossing::Through);
        assert_eq!(g.crossing(b, a, 0.2), Crossing::None, "backwards");
        let (a, b) = p(0.95, 2.0);
        assert_eq!(g.crossing(a, b, 0.2), Crossing::Frame);
        let (a, b) = p(5.0, 2.0);
        assert_eq!(g.crossing(a, b, 0.2), Crossing::Missed);
    }

    #[test]
    fn circle_gates_face_along_the_loop() {
        let t = Track::circle("c", 4, 10.0, 2.0, 2.0, 1.0);
        for (k, g) in t.gates.iter().enumerate() {
            let next = t.gates[(k + 1) % 4].center();
            assert!((next - g.center()).dot(g.normal()) > 0.0);
        }
    }

    #[test]
    fn times_laps_from_gate_zero() {
        let mut lt = LapTimer::default();
        assert_eq!(lt.pass(3, 1.0), None); // start
        assert_eq!(lt.pass(3, 2.0), None);
        assert_eq!(lt.pass(3, 3.0), None);
        assert_eq!(lt.pass(3, 4.5), Some(3.5));
        assert_eq!(lt.pass(3, 5.0), None);
        assert_eq!(lt.pass(3, 6.0), None);
        assert_eq!(lt.pass(3, 7.0), Some(2.5));
        assert_eq!(lt.best_lap(), Some(2.5));
    }

    #[test]
    fn leaderboard_ranks_and_persists() {
        let rec = |t: f64| LapRecord {
            pilot: "p".into(),
            vehicle: "v".into(),
            track: "t".into(),
            lap_time: t,
            seed: 0,
        };
        let mut lb = Leaderboard::new(2);
        assert_eq!(lb.submit(rec(5.0)), Some(1));
        assert_eq!(lb.submit(rec(4.0)), Some(1));
        assert_eq!(lb.submit(rec(9.0)), None);
        assert_eq!(lb.submit(rec(4.5)), Some(2));
        assert_eq!(
            lb.entries.iter().map(|e| e.lap_time).collect::<Vec<_>>(),
            [4.0, 4.5]
        );

        let dir = std::env::temp_dir().join(format!("dronelab-lb-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("board.json");
        lb.save(&path).unwrap();
        assert_eq!(Leaderboard::load_or_new(&path, 2).unwrap(), lb);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}

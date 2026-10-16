//! Lithium battery pack with state of charge and voltage sag.

/// Open-circuit cell voltage at 0%, 10%, ... 100% charge (typical Li-ion/LiPo).
const OCV: [f32; 11] = [
    3.27, 3.65, 3.72, 3.76, 3.80, 3.84, 3.88, 3.94, 4.02, 4.10, 4.20,
];

/// Nominal cell voltage, V. Rotor speed limits are quoted at this voltage.
pub const CELL_NOMINAL: f32 = 3.85;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BatterySpec {
    /// Cells in series.
    pub cells: u32,
    pub capacity_ah: f32,
    /// Pack internal resistance including wiring, ohm.
    pub r_internal: f32,
    /// Constant draw of avionics, cooling etc., W.
    pub aux_power: f32,
}

impl BatterySpec {
    pub fn nominal_voltage(&self) -> f32 {
        self.cells as f32 * CELL_NOMINAL
    }

    pub fn open_circuit_voltage(&self, soc: f32) -> f32 {
        let x = soc.clamp(0.0, 1.0) * 10.0;
        let i = (x as usize).min(9);
        let f = x - i as f32;
        self.cells as f32 * (OCV[i] + f * (OCV[i + 1] - OCV[i]))
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Battery {
    pub spec: BatterySpec,
    /// State of charge, 0..1.
    pub soc: f32,
    /// Terminal voltage under the present load, V.
    pub voltage: f32,
    pub current: f32,
}

impl Battery {
    pub fn new(spec: BatterySpec, soc: f32) -> Self {
        let soc = soc.clamp(0.0, 1.0);
        Battery {
            spec,
            soc,
            voltage: spec.open_circuit_voltage(soc),
            current: 0.0,
        }
    }

    /// Draws `power` watts for `dt` seconds and updates voltage and charge.
    ///
    /// Terminal voltage solves `V = OCV - I R` with `I = P / V`. If the pack
    /// cannot deliver `power` at all, it delivers its maximum (`V = OCV/2`).
    pub fn draw(&mut self, power: f32, dt: f32) {
        let ocv = self.spec.open_circuit_voltage(self.soc);
        let r = self.spec.r_internal;
        let disc = ocv * ocv - 4.0 * (power + self.spec.aux_power) * r;
        self.voltage = if disc > 0.0 {
            0.5 * (ocv + libm::sqrtf(disc))
        } else {
            0.5 * ocv
        };
        self.current = (ocv - self.voltage) / r.max(1e-6);
        self.soc = (self.soc - self.current * dt / (3600.0 * self.spec.capacity_ah)).max(0.0);
    }

    /// Terminal voltage over nominal voltage; scales the top rotor speed.
    pub fn voltage_ratio(&self) -> f32 {
        self.voltage / self.spec.nominal_voltage()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPEC: BatterySpec = BatterySpec {
        cells: 96,
        capacity_ah: 34.0,
        r_internal: 0.2,
        aux_power: 0.0,
    };

    #[test]
    fn sags_under_load_and_drains() {
        let mut b = Battery::new(SPEC, 1.0);
        let rest = b.voltage;
        b.draw(30_000.0, 1.0);
        assert!(b.voltage < rest - 10.0, "{} vs {rest}", b.voltage);
        assert!((b.current * b.voltage - 30_000.0).abs() < 1.0);
        let soc = b.soc;
        for _ in 0..600 {
            b.draw(30_000.0, 1.0);
        }
        assert!(b.soc < soc - 0.3);
    }

    #[test]
    fn collapses_instead_of_nan() {
        let mut b = Battery::new(SPEC, 0.05);
        b.draw(1.0e7, 0.01);
        assert!(b.voltage.is_finite() && b.voltage > 0.0);
    }
}

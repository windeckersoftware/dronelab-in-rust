/// Drone models enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DroneModel {
    /// Bitcraze Crazyflie 2.0 in the X configuration.
    Cf2x,
    /// Bitcraze Crazyflie 2.0 in the + configuration.
    Cf2p,
    /// Racer drone in the X configuration.
    Race,
}

impl DroneModel {
    /// The model's name, also the stem of its `.urdf` file in `assets/`.
    pub fn value(self) -> &'static str {
        match self {
            DroneModel::Cf2x => "cf2x",
            DroneModel::Cf2p => "cf2p",
            DroneModel::Race => "racer",
        }
    }

    /// The bundled URDF description of this model.
    pub fn urdf(self) -> &'static str {
        match self {
            DroneModel::Cf2x => include_str!("../assets/cf2x.urdf"),
            DroneModel::Cf2p => include_str!("../assets/cf2p.urdf"),
            DroneModel::Race => include_str!("../assets/racer.urdf"),
        }
    }
}

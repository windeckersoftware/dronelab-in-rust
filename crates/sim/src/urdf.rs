//! A small reader for the custom parameters in the drones' `.urdf` files.

use std::fmt;

use roxmltree::{Document, Node};

/// A parameter that can be read from a drone's URDF file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UrdfParam {
    /// Mass, from `<link><inertial><mass value>`.
    M,
    Ixx,
    Iyy,
    Izz,
    Arm,
    Thrust2Weight,
    Kf,
    Km,
    MaxSpeedKmh,
    GndEffCoeff,
    PropRadius,
    DragCoeffXy,
    DragCoeffZ,
    DwCoeff1,
    DwCoeff2,
    DwCoeff3,
    /// Collision cylinder length.
    Length,
    /// Collision cylinder radius.
    Radius,
    /// z component of the collision shape's origin.
    CollisionZOffset,
}

impl UrdfParam {
    /// The XML attribute name, as used in the Python `_getURDFParameter`.
    pub fn name(self) -> &'static str {
        match self {
            UrdfParam::M => "m",
            UrdfParam::Ixx => "ixx",
            UrdfParam::Iyy => "iyy",
            UrdfParam::Izz => "izz",
            UrdfParam::Arm => "arm",
            UrdfParam::Thrust2Weight => "thrust2weight",
            UrdfParam::Kf => "kf",
            UrdfParam::Km => "km",
            UrdfParam::MaxSpeedKmh => "max_speed_kmh",
            UrdfParam::GndEffCoeff => "gnd_eff_coeff",
            UrdfParam::PropRadius => "prop_radius",
            UrdfParam::DragCoeffXy => "drag_coeff_xy",
            UrdfParam::DragCoeffZ => "drag_coeff_z",
            UrdfParam::DwCoeff1 => "dw_coeff_1",
            UrdfParam::DwCoeff2 => "dw_coeff_2",
            UrdfParam::DwCoeff3 => "dw_coeff_3",
            UrdfParam::Length => "length",
            UrdfParam::Radius => "radius",
            UrdfParam::CollisionZOffset => "collision_z_offset",
        }
    }
}

#[derive(Debug)]
pub enum UrdfError {
    Xml(roxmltree::Error),
    /// An element on the path to the parameter is missing.
    MissingElement(&'static str),
    MissingAttribute {
        element: &'static str,
        attribute: &'static str,
    },
    InvalidNumber {
        attribute: &'static str,
        value: String,
    },
}

impl fmt::Display for UrdfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UrdfError::Xml(e) => write!(f, "invalid URDF XML: {e}"),
            UrdfError::MissingElement(path) => write!(f, "URDF has no <{path}> element"),
            UrdfError::MissingAttribute { element, attribute } => {
                write!(f, "URDF <{element}> has no `{attribute}` attribute")
            }
            UrdfError::InvalidNumber { attribute, value } => {
                write!(f, "URDF attribute `{attribute}` is not a number: {value:?}")
            }
        }
    }
}

impl std::error::Error for UrdfError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            UrdfError::Xml(e) => Some(e),
            _ => None,
        }
    }
}

impl From<roxmltree::Error> for UrdfError {
    fn from(e: roxmltree::Error) -> Self {
        UrdfError::Xml(e)
    }
}

/// Reads a parameter from a drone's URDF text.
///
/// The Python version walks the tree by position (`URDF_TREE[1][2][1][0]`);
/// this one follows the same path by tag name, which gives the same result
/// for the bundled files without depending on element order.
pub fn get_urdf_parameter(urdf: &str, param: UrdfParam) -> Result<f64, UrdfError> {
    let doc = Document::parse(urdf)?;
    let robot = doc.root_element();
    let base_link = child(robot, "link")?;
    let collision = || child(base_link, "collision");

    match param {
        UrdfParam::M => attr_f64(
            child(child(base_link, "inertial")?, "mass")?,
            "mass",
            "value",
        ),
        UrdfParam::Ixx | UrdfParam::Iyy | UrdfParam::Izz => attr_f64(
            child(child(base_link, "inertial")?, "inertia")?,
            "inertia",
            param.name(),
        ),
        UrdfParam::Arm
        | UrdfParam::Thrust2Weight
        | UrdfParam::Kf
        | UrdfParam::Km
        | UrdfParam::MaxSpeedKmh
        | UrdfParam::GndEffCoeff
        | UrdfParam::PropRadius
        | UrdfParam::DragCoeffXy
        | UrdfParam::DragCoeffZ
        | UrdfParam::DwCoeff1
        | UrdfParam::DwCoeff2
        | UrdfParam::DwCoeff3 => attr_f64(child(robot, "properties")?, "properties", param.name()),
        UrdfParam::Length | UrdfParam::Radius => {
            let geometry = child(collision()?, "geometry")?;
            let shape = geometry
                .children()
                .find(Node::is_element)
                .ok_or(UrdfError::MissingElement("collision><geometry><*"))?;
            attr_f64(shape, "collision geometry shape", param.name())
        }
        UrdfParam::CollisionZOffset => {
            let origin = child(collision()?, "origin")?;
            let xyz = attr(origin, "origin", "xyz")?;
            let z = xyz
                .split_whitespace()
                .nth(2)
                .ok_or_else(|| UrdfError::InvalidNumber {
                    attribute: "xyz",
                    value: xyz.to_string(),
                })?;
            parse_f64("xyz", z)
        }
    }
}

/// Hub positions of the `prop<N>_link` links, in file order, from each
/// link's `<inertial><origin xyz>`.
pub fn rotor_positions(urdf: &str) -> Result<Vec<(f64, f64, f64)>, UrdfError> {
    let doc = Document::parse(urdf)?;
    let props = doc.root_element().children().filter(|n| {
        n.has_tag_name("link")
            && n.attribute("name")
                .is_some_and(|name| name.starts_with("prop") && name.ends_with("_link"))
    });
    props
        .map(|link| {
            let origin = child(child(link, "inertial")?, "origin")?;
            let xyz = attr(origin, "origin", "xyz")?;
            let v: Vec<f64> = xyz
                .split_whitespace()
                .map(|s| parse_f64("xyz", s))
                .collect::<Result<_, _>>()?;
            match v[..] {
                [x, y, z] => Ok((x, y, z)),
                _ => Err(UrdfError::InvalidNumber {
                    attribute: "xyz",
                    value: xyz.to_string(),
                }),
            }
        })
        .collect()
}

fn child<'a, 'input>(
    node: Node<'a, 'input>,
    tag: &'static str,
) -> Result<Node<'a, 'input>, UrdfError> {
    node.children()
        .find(|n| n.has_tag_name(tag))
        .ok_or(UrdfError::MissingElement(tag))
}

fn attr<'a>(
    node: Node<'a, '_>,
    element: &'static str,
    attribute: &'static str,
) -> Result<&'a str, UrdfError> {
    node.attribute(attribute)
        .ok_or(UrdfError::MissingAttribute { element, attribute })
}

fn attr_f64(
    node: Node<'_, '_>,
    element: &'static str,
    attribute: &'static str,
) -> Result<f64, UrdfError> {
    parse_f64(attribute, attr(node, element, attribute)?)
}

fn parse_f64(attribute: &'static str, value: &str) -> Result<f64, UrdfError> {
    value.trim().parse().map_err(|_| UrdfError::InvalidNumber {
        attribute,
        value: value.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DroneModel;

    fn get(model: DroneModel, p: UrdfParam) -> f64 {
        get_urdf_parameter(model.urdf(), p).unwrap()
    }

    #[test]
    fn reads_cf2x_parameters() {
        let m = DroneModel::Cf2x;
        assert_eq!(get(m, UrdfParam::M), 0.027);
        assert_eq!(get(m, UrdfParam::Ixx), 1.4e-5);
        assert_eq!(get(m, UrdfParam::Izz), 2.17e-5);
        assert_eq!(get(m, UrdfParam::Kf), 3.16e-10);
        assert_eq!(get(m, UrdfParam::Km), 7.94e-12);
        assert_eq!(get(m, UrdfParam::DwCoeff2), 0.16);
        assert_eq!(get(m, UrdfParam::DwCoeff3), -0.11);
        assert_eq!(get(m, UrdfParam::Length), 0.025);
        assert_eq!(get(m, UrdfParam::Radius), 0.06);
        assert_eq!(get(m, UrdfParam::CollisionZOffset), 0.0);
    }

    #[test]
    fn reads_racer_parameters() {
        let m = DroneModel::Race;
        assert_eq!(get(m, UrdfParam::M), 0.830);
        assert_eq!(get(m, UrdfParam::Arm), 0.109);
        assert_eq!(get(m, UrdfParam::MaxSpeedKmh), 200.0);
    }

    #[test]
    fn missing_attribute_is_an_error() {
        let urdf = r#"<robot><properties/><link/></robot>"#;
        let err = get_urdf_parameter(urdf, UrdfParam::Kf).unwrap_err();
        assert!(matches!(
            err,
            UrdfError::MissingAttribute {
                attribute: "kf",
                ..
            }
        ));
    }
}

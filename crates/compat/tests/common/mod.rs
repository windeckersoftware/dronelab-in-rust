//! Loading recorded upstream data and comparing numbers.
#![allow(dead_code)]

use serde_json::Value;

pub fn golden(name: &str) -> Value {
    let path = format!("{}/tests/golden/{name}.json", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let v: Value = serde_json::from_str(&text).unwrap();
    v["data"].clone()
}

pub fn f(v: &Value) -> f64 {
    v.as_f64().unwrap_or_else(|| panic!("not a number: {v}"))
}

pub fn vec(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(f).collect()
}

pub fn v3(v: &Value) -> [f64; 3] {
    let x = vec(v);
    [x[0], x[1], x[2]]
}

pub fn v4(v: &Value) -> [f64; 4] {
    let x = vec(v);
    [x[0], x[1], x[2], x[3]]
}

pub fn mat3(v: &Value) -> [[f64; 3]; 3] {
    let rows = v.as_array().unwrap();
    [v3(&rows[0]), v3(&rows[1]), v3(&rows[2])]
}

/// Flattens nested arrays of numbers.
pub fn flat(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().flat_map(flat).collect(),
        Value::Bool(b) => vec![*b as u8 as f64],
        other => vec![f(other)],
    }
}

/// Largest |a - b| / max(1, |b|) over two equal-length slices.
pub fn max_rel_err(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs() / y.abs().max(1.0))
        .fold(0.0, f64::max)
}

#[track_caller]
pub fn assert_close(what: &str, a: &[f64], b: &[f64], tol: f64) {
    let e = max_rel_err(a, b);
    assert!(
        e <= tol,
        "{what}: max relative error {e:e} > {tol:e}\n ours     {:?}\n upstream {:?}",
        &a[..a.len().min(8)],
        &b[..b.len().min(8)]
    );
}

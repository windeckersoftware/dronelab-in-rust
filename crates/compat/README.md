# dronelab-compat

An `f64`, line-by-line Rust port of
[gym-pybullet-drones](https://github.com/learnsyslab/gym-pybullet-drones)
v2.2.0, commit `01563120e060c9233a61f9cf00e6d4e5a8b331ab`, checked against
outputs recorded from that exact code. The repository README has the
measured agreement and the list of what is and isn't reproduced.

| module | ports |
|---|---|
| `math` | `p.getMatrixFromQuaternion`, `p.getEulerFromQuaternion`, `p.getQuaternionFromEuler`, Bullet's matrix-to-quaternion read-back, SciPy `Rotation` with `'XYZ'`, transforms3d quaternions |
| `params` | `BaseAviary._parseURDFParameters` and the derived constants |
| `control` | `DSLPIDControl`, `CTBRControl`, `MRACControl` (reference-model matrices from upstream in `mrac_tables.rs`) |
| `physics` | `_physics`, `_groundEffect`, `_drag`, `_downwash`, `_dynamics`, `_integrateQ`, plus a stand-in for PyBullet's rigid-body step |
| `aviary` | `BaseAviary.step`/`reset`, `CtrlAviary`, `VelocityAviary`, `BaseRLAviary` (all action types, KIN observations), `HoverAviary`, `MultiHoverAviary` |
| `batch` | many aviaries in parallel with automatic reset |

## Reference data

`tests/golden/*.json` was written by `tools/record_upstream.py`, which runs
upstream itself and stores the inputs it chose plus everything upstream
returned. It also wraps PyBullet's `applyExternalForce`/`applyExternalTorque`,
so the tests can check the exact forces upstream feeds the engine. To
re-record, from the repository root, with upstream installed in
`bench/.venv-pybullet`:

```bash
bench/.venv-pybullet/bin/python crates/compat/tools/record_upstream.py            # everything
bench/.venv-pybullet/bin/python crates/compat/tools/record_upstream.py envs       # one section
bench/.venv-pybullet/bin/python crates/compat/tools/gen_mrac_tables.py            # src/mrac_tables.rs
```

The recordings name the upstream commit and the PyBullet and NumPy versions
they came from.

"""Flight log export: write a short eVTOL flight and read it back with the
PX4 (pyulog) and ArduPilot (pymavlink) readers."""

import math

import numpy as np
import pytest

from dronelab import BatchSim

pyulog = pytest.importorskip("pyulog")
DFReader = pytest.importorskip("pymavlink.DFReader")

STEPS = 90
ORIGIN = (47.0, 19.0, 120.0)


@pytest.fixture(scope="module")
def flight(tmp_path_factory):
    """Env 1 of 2 flies to its target with the PID expert; returns the log
    paths and the states seen from Python at each recorded sample."""
    sim = BatchSim(num_envs=2, airframe="evtol_single_seat", action="velocity", seed=5)
    sim.record(env_index=1, origin=ORIGIN)
    states = [sim.state()]
    for _ in range(STEPS):
        _, _, term, trunc, _ = sim.step(sim.expert_actions())
        assert not (term[1] or trunc[1])
        states.append(sim.state())
    out = tmp_path_factory.mktemp("logs")
    paths = {"ulg": out / "flight.ulg", "bin": out / "flight.bin"}
    for fmt, path in paths.items():
        assert sim.save_log(str(path), env_index=1) == STEPS + 1
    pick = lambda key: np.array([s[key][1] for s in states])
    return paths, {k: pick(k) for k in ("pos", "vel", "rpy", "battery_voltage", "battery_soc", "time")}


def test_ulog_round_trip(flight):
    paths, st = flight
    ulog = pyulog.ULog(str(paths["ulg"]))
    n = STEPS + 1
    topics = {d.name: d for d in ulog.data_list}
    for name in (
        "vehicle_attitude",
        "vehicle_angular_velocity",
        "sensor_combined",
        "vehicle_local_position",
        "vehicle_global_position",
        "battery_status",
        "actuator_outputs",
    ):
        assert len(topics[name].data["timestamp"]) == n, name

    lp = topics["vehicle_local_position"].data
    np.testing.assert_allclose(-lp["z"], st["pos"][:, 2], atol=1e-4)  # NED down = -up
    np.testing.assert_allclose(lp["x"], st["pos"][:, 1], atol=1e-4)  # north = y
    np.testing.assert_allclose(lp["vy"], st["vel"][:, 0], atol=1e-4)  # east = x
    np.testing.assert_allclose(lp["timestamp"] / 1e6, st["time"], atol=1e-6)
    assert lp["ref_lat"][0] == ORIGIN[0]

    gp = topics["vehicle_global_position"].data
    np.testing.assert_allclose(gp["alt"], ORIGIN[2] + st["pos"][:, 2], atol=1e-3)

    bat = topics["battery_status"].data
    np.testing.assert_allclose(bat["voltage_v"], st["battery_voltage"], rtol=1e-6)
    np.testing.assert_allclose(bat["remaining"], st["battery_soc"], rtol=1e-6)
    assert bat["discharged_mah"][-1] > 0

    att = topics["vehicle_attitude"].data
    q = np.stack([att[f"q[{i}]"] for i in range(4)], axis=1)
    w, x, y, z = q.T
    heading = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    expected = np.angle(np.exp(1j * (np.pi / 2 - st["rpy"][:, 2])))
    np.testing.assert_allclose(np.angle(np.exp(1j * (heading - expected))), 0, atol=1e-4)

    out = topics["actuator_outputs"].data
    assert set(out["noutputs"]) == {8}
    pwm = np.stack([out[f"output[{i}]"] for i in range(8)], axis=1)
    assert pwm.min() >= 1000 and pwm.max() <= 2000 and pwm.mean() > 1100
    assert not out["output[8]"].any()

    sc = topics["sensor_combined"].data
    # Hovering: the FRD accelerometer reads about -g on z.
    assert abs(np.median(sc["accelerometer_m_s2[2]"]) + 9.81) < 1.0

    assert ulog.initial_parameters["BAT1_N_CELLS"] == 96
    assert ulog.msg_info_dict["sys_name"] == "dronelab"
    assert any("episode 0 seed 6" in m.message for m in ulog.logged_messages)


def test_dataflash_round_trip(flight):
    paths, st = flight
    log = DFReader.DFReader_binary(str(paths["bin"]), zero_time_base=True)
    msgs = {}
    while (m := log.recv_msg()) is not None:
        msgs.setdefault(m.get_type(), []).append(m)
    n = STEPS + 1
    for name in ("ATT", "IMU", "GPS", "POS", "BAT", "RCOU", "XKF1"):
        assert len(msgs[name]) == n, name
    assert len(msgs["ESC"]) == 8 * n
    assert "RCO2" not in msgs
    assert {p.Name for p in msgs["PARM"]} == {"BATT_CAPACITY", "BATT_MONITOR", "FRAME_CLASS"}
    assert any("episode 0 seed 6" in m.Message for m in msgs["MSG"])

    alt = st["pos"][:, 2]
    rel = np.array([m.RelHomeAlt for m in msgs["POS"]])
    np.testing.assert_allclose(rel, alt, atol=1e-3)
    gps_alt = np.array([m.Alt for m in msgs["GPS"]])  # cm resolution
    np.testing.assert_allclose(gps_alt, ORIGIN[2] + alt, atol=0.006)
    pd = np.array([m.PD for m in msgs["XKF1"]])
    np.testing.assert_allclose(pd, -alt, atol=1e-4)
    lat = np.array([m.Lat for m in msgs["POS"]])
    north = np.radians(lat - ORIGIN[0]) * 6378137.0
    np.testing.assert_allclose(north, st["pos"][:, 1], atol=0.02)  # 1e-7 deg ~ 1 cm

    volt = np.array([m.Volt for m in msgs["BAT"]])
    np.testing.assert_allclose(volt, st["battery_voltage"], rtol=1e-6)
    assert msgs["BAT"][-1].CurrTot > 0

    yaw = np.array([m.Yaw for m in msgs["ATT"]])
    expected = np.degrees(np.pi / 2 - st["rpy"][:, 2]) % 360
    diff = (yaw - expected + 180) % 360 - 180
    assert np.abs(diff).max() < 0.011  # centidegrees
    roll = np.array([m.Roll for m in msgs["ATT"]])
    pitch = np.array([m.Pitch for m in msgs["ATT"]])
    np.testing.assert_allclose(roll, np.degrees(st["rpy"][:, 0]), atol=0.011)
    np.testing.assert_allclose(pitch, -np.degrees(st["rpy"][:, 1]), atol=0.011)

    c1 = np.array([m.C1 for m in msgs["RCOU"]])
    assert c1.min() >= 1000 and c1.max() <= 2000
    assert all(m.C9 == 0 for m in msgs["RCOU"])

    # Timestamps from TimeUS, 30 Hz.
    t = np.array([m.TimeUS for m in msgs["ATT"]]) / 1e6
    np.testing.assert_allclose(np.diff(t), 1 / 30, atol=1e-6)


def test_bad_log_requests_raise(tmp_path):
    sim = BatchSim(num_envs=1, airframe="racer")
    with pytest.raises(ValueError, match="not recording"):
        sim.save_log(str(tmp_path / "x.ulg"))
    with pytest.raises(IndexError):
        sim.record(env_index=3)
    sim.record()
    with pytest.raises(ValueError, match="format"):
        sim.save_log(str(tmp_path / "x.csv"))
    sim.stop_recording()
    with pytest.raises(ValueError):
        sim.save_log(str(tmp_path / "x.bin"))


def test_recording_spans_auto_reset(tmp_path):
    sim = BatchSim(num_envs=1, airframe="crazyflie_x", episode_seconds=0.5, control_hz=30)
    sim.record()
    for _ in range(40):  # 15-step episodes: two auto-resets
        sim.step(sim.expert_actions())
    ulog_path = tmp_path / "multi.ulg"
    assert sim.save_log(str(ulog_path)) == 1 + 40 + 2
    ulog = pyulog.ULog(str(ulog_path))
    t = ulog.get_dataset("vehicle_local_position").data["timestamp"]
    assert np.all(np.diff(t) > 0)
    assert sum("episode" in m.message for m in ulog.logged_messages) == 3

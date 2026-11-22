import numpy as np
import pytest

from simulation.env_core import NavEnv
from simulation.sensors import EKF, SensorNoise, lidar_scan


def test_lidar_known_geometry():
    pos = np.array([10.0, 10.0])
    obs = np.array([[20.0, 10.0]])
    r = lidar_scan(pos, 0.0, obs, np.array([1.0]), n_beams=4, max_range=100, world=64)
    # beam 0 -> +x hits circle at 9; beam 1 -> +y hits wall at 54;
    # beam 2 -> -x wall at 10; beam 3 -> -y wall at 10
    assert r == pytest.approx([9.0, 54.0, 10.0, 10.0])


def test_lidar_respects_heading_and_range():
    pos = np.array([10.0, 10.0])
    obs = np.array([[10.0, 20.0]])
    r = lidar_scan(pos, np.pi / 2, obs, np.array([1.0]), n_beams=4, max_range=15, world=64)
    assert r[0] == pytest.approx(9.0)        # beam 0 now points +y
    assert r[2] == pytest.approx(10.0)       # beam 2 points -y, wall at y=0
    assert r[3] == pytest.approx(15.0)       # +x wall is 54 away, capped at max range


def test_lidar_inside_obstacle_reads_zero():
    r = lidar_scan(np.array([5.0, 5.0]), 0.0, np.array([[5.0, 5.0]]), np.array([2.0]), 8, 15, 64)
    assert np.allclose(r, 0.0)


def test_ekf_heading_wraparound():
    """Drive in a circle so heading crosses +-pi many times; the estimate must
    track it. Without wrapping the innovation the error jumps to ~2pi."""
    env = NavEnv(0.0, noise=SensorNoise(gps_degraded_p=0.0))
    env.reset(1)
    env.obstacles = []
    env._refresh_obstacle_arrays()
    env.state = np.array([32.0, 32.0, 3.0, 2.0])
    env.sensors.reset(env.state)
    env.goal = np.array([1000.0, 1000.0])   # never reached
    env.dp.max_steps = 10_000
    errs = []
    for _ in range(300):
        m, *_ = env.step([0.0, 0.6])
        errs.append(abs((m.est[2] - env.state[2] + np.pi) % (2 * np.pi) - np.pi))
    assert max(errs) < 0.3


def test_ekf_beats_raw_gps():
    env = NavEnv(0.0)
    env.reset(7)
    env.obstacles = []
    env._refresh_obstacle_arrays()
    env.dp.max_steps = 10_000
    gps_e, ekf_e = [], []
    for t in range(150):
        m, _, done, _ = env.step([0.2, 0.15 * np.sin(t / 10)])
        if done:
            break
        gps_e.append(np.linalg.norm(m.gps - env.state[:2]))
        ekf_e.append(np.linalg.norm(m.est[:2] - env.state[:2]))
    assert np.mean(ekf_e) < 0.7 * np.mean(gps_e)


def test_gps_gate_rejects_outlier():
    f = EKF(2.0, 0.2, 4.0)
    f.reset([10, 10, 0, 1], np.diag([0.1, 0.1, 0.01, 0.01]))
    f.update_gps(np.array([40.0, 40.0]), sigma=0.5)
    assert f.gps_rejected == 1
    assert np.allclose(f.x[:2], [10, 10])


def test_noise_scaling():
    n = SensorNoise().scaled(2.0)
    assert n.gps_sigma == pytest.approx(2 * SensorNoise().gps_sigma)
    assert n.lidar_dropout == SensorNoise().lidar_dropout

import numpy as np
import pytest

from simulation.env_core import NavEnv, DifficultyParams, Obstacle, wrap_angle


def test_difficulty_endpoints():
    lo, hi = DifficultyParams.from_level(0.0), DifficultyParams.from_level(1.0)
    assert (lo.n_static, lo.n_dynamic, lo.max_steps) == (3, 0, 150)
    assert (hi.n_static, hi.n_dynamic, hi.max_steps) == (14, 4, 400)
    assert lo.goal_dist == pytest.approx(35.0) and hi.goal_dist == pytest.approx(57.0)


@pytest.mark.parametrize("d", [0.0, 0.5, 1.0])
def test_start_goal_distance_matches_curriculum(d):
    # regression: the old rejection sampler under-shot the distance at d=1
    env = NavEnv(d)
    for s in range(30):
        env.reset(s)
        dist = np.linalg.norm(env.goal - env.start)
        assert dist == pytest.approx(env.dp.goal_dist, abs=1e-6)
        assert 0 < env.goal[0] < env.WORLD and 0 < env.goal[1] < env.WORLD


def test_obstacles_clear_of_start_and_goal():
    env = NavEnv(1.0)
    for s in range(30):
        env.reset(s)
        for o in env.obstacles:
            assert np.hypot(o.x - env.start[0], o.y - env.start[1]) > o.r + env.vp.radius
            assert np.hypot(o.x - env.goal[0], o.y - env.goal[1]) > o.r


def test_reset_is_deterministic_per_seed():
    a, b = NavEnv(1.0), NavEnv(1.0)
    ma, mb = a.reset(123), b.reset(123)
    assert np.allclose(a.obs_xy, b.obs_xy)
    assert np.allclose(ma.lidar, mb.lidar) and np.allclose(ma.gps, mb.gps)
    for _ in range(20):
        ma, *_ = a.step([0.5, 0.1])
        mb, *_ = b.step([0.5, 0.1])
    assert np.allclose(ma.est, mb.est)


def _empty_env():
    env = NavEnv(0.0)
    env.reset(0)
    env.obstacles = []
    env._refresh_obstacle_arrays()
    return env


def test_collision_with_obstacle():
    env = _empty_env()
    x, y, th, _ = env.state
    env.obstacles = [Obstacle(x + 3 * np.cos(th), y + 3 * np.sin(th), 1.0)]
    env._refresh_obstacle_arrays()
    for _ in range(50):
        _, r, done, info = env.step([1.0, 0.0])
        if done:
            break
    assert info.event == "collision" and info.collided_with == "static"
    assert r < -10


def test_leaving_world_is_a_collision():
    env = _empty_env()
    env.state = np.array([2.0, 30.0, np.pi, 3.0])
    env.goal = np.array([60.0, 30.0])
    for _ in range(10):
        _, _, done, info = env.step([1.0, 0.0])
        if done:
            break
    assert info.event == "collision" and info.collided_with == "wall"


def test_goal_reached():
    env = _empty_env()
    x, y, th, _ = env.state
    env.goal = np.array([x + 4 * np.cos(th), y + 4 * np.sin(th)])
    env.prev_dist = 4.0
    for _ in range(30):
        _, r, done, info = env.step([1.0, 0.0])
        if done:
            break
    assert info.event == "goal" and r > 10


def test_idle_penalty_escalates():
    env = _empty_env()
    rs = []
    for _ in range(10):
        _, r, _, _ = env.step([-1.0, 0.0])   # stay parked
        rs.append(r)
    assert rs[-1] < rs[0] < 0


def test_wrap_angle():
    assert wrap_angle(np.pi + 0.1) == pytest.approx(-np.pi + 0.1)
    assert wrap_angle(-np.pi - 0.1) == pytest.approx(np.pi - 0.1)
    assert wrap_angle(0.3) == pytest.approx(0.3)

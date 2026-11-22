import numpy as np

from simulation.env_core import NavEnv
from navigation.classical_planners import GridAStar, RRT, PlannerAgent, segment_clear, shortcut


def test_segment_clear():
    c, r = np.array([[5.0, 0.0]]), np.array([1.0])
    assert not segment_clear(np.array([0.0, 0.0]), np.array([10.0, 0.0]), c, r)
    assert segment_clear(np.array([0.0, 2.0]), np.array([10.0, 2.0]), c, r)


def test_astar_empty_world_is_straight():
    a = GridAStar(64.0)
    occ = a.occupancy(np.zeros((0, 2)), np.zeros(0), 1.0)
    p = a.plan(np.array([5.0, 5.0]), np.array([50.0, 50.0]), occ)
    p = shortcut(p, np.zeros((0, 2)), np.zeros(0))
    assert len(p) == 2


def test_astar_goes_around_obstacle():
    a = GridAStar(64.0)
    c, r = np.array([[30.0, 30.0]]), np.array([6.0])
    p = a.plan(np.array([10.0, 30.0]), np.array([50.0, 30.0]), a.occupancy(c, r, 1.0))
    assert p is not None
    assert np.min(np.linalg.norm(p - c[0], axis=1)) >= 6.0 - a.res


def test_astar_returns_none_when_enclosed():
    a = GridAStar(64.0)
    ang = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    ring = np.stack([50 + 6 * np.cos(ang), 50 + 6 * np.sin(ang)], 1)
    occ = a.occupancy(ring, np.full(40, 1.5), 1.0)
    assert a.plan(np.array([5.0, 5.0]), np.array([50.0, 50.0]), occ) is None


def test_rrt_finds_collision_free_path():
    c, r = np.array([[30.0, 30.0]]), np.array([6.0])
    p = RRT(64.0, rng=np.random.default_rng(0)).plan(
        np.array([10.0, 30.0]), np.array([50.0, 30.0]), c, r, 1.0)
    assert p is not None
    for i in range(len(p) - 1):
        assert segment_clear(p[i], p[i + 1], c, r)


def test_replanning_astar_solves_easy_episodes():
    env = NavEnv(0.0)
    ag = PlannerAgent("astar", replan_every=10)
    wins = 0
    for s in range(10):
        m = env.reset(9000 + s)
        ag.reset(env, m, s)
        done = False
        while not done:
            m, _, done, info = env.step(ag.act(m))
        wins += info.event == "goal"
    assert wins >= 8

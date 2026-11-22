"""
2D continuous navigation environment.

A single vehicle (kinematic bicycle model) has to reach a goal in a square world
with circular static obstacles and circular obstacles that move at constant
velocity and bounce off the walls. Leaving the world counts as a collision.

The env itself only knows ground truth. Sensors + EKF live in sensors.py and are
attached by the env so that every consumer (PPO, planners, benchmark) sees the
exact same noisy measurements for a given seed.

Difficulty d in [0, 1] scales the task (see `DifficultyParams.from_level`):
    static obstacles   3 -> 14
    dynamic obstacles  0 -> 4
    max steps          150 -> 400
    start-goal dist    35 -> 57
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from simulation.sensors import SensorSuite, SensorNoise


def wrap_angle(a):
    """Wrap angle(s) to (-pi, pi]."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


@dataclass
class VehicleParams:
    wheelbase: float = 2.0
    radius: float = 0.8          # collision radius
    max_speed: float = 4.0
    max_accel: float = 2.0
    max_steer: float = 0.6       # rad
    dt: float = 0.2


@dataclass
class DifficultyParams:
    n_static: int
    n_dynamic: int
    max_steps: int
    goal_dist: float

    @classmethod
    def from_level(cls, d: float) -> "DifficultyParams":
        d = float(np.clip(d, 0.0, 1.0))
        return cls(
            n_static=int(round(3 + 11 * d)),
            n_dynamic=int(round(4 * d)),
            max_steps=int(round(150 + 250 * d)),
            goal_dist=35.0 + 22.0 * d,
        )


@dataclass
class Obstacle:
    x: float
    y: float
    r: float
    vx: float = 0.0
    vy: float = 0.0

    @property
    def dynamic(self) -> bool:
        return self.vx != 0.0 or self.vy != 0.0


@dataclass
class RewardConfig:
    progress: float = 1.0          # per unit of distance closed
    time_cost: float = 0.01
    goal_bonus: float = 20.0
    collision_penalty: float = 20.0
    prox_radius: float = 2.0       # clearance below which the proximity term kicks in
    prox_max: float = 0.05         # cap per step, kept well below progress scale (~0.8/step)
    idle_speed: float = 0.3
    idle_base: float = 0.005       # escalating: idle_base * consecutive_idle_steps
    idle_cap: float = 0.1


@dataclass
class StepInfo:
    event: str = ""                # "", "goal", "collision", "timeout"
    min_clearance: float = float("inf")
    path_length: float = 0.0
    collided_with: str = ""        # "static" | "dynamic" | "wall"


class NavEnv:
    WORLD = 64.0
    GOAL_RADIUS = 2.0

    def __init__(
        self,
        difficulty: float = 1.0,
        vehicle: VehicleParams | None = None,
        reward: RewardConfig | None = None,
        noise: SensorNoise | None = None,
        n_beams: int = 36,
        lidar_range: float = 15.0,
    ):
        self.vp = vehicle or VehicleParams()
        self.rc = reward or RewardConfig()
        self.noise = noise or SensorNoise()
        self.n_beams = n_beams
        self.lidar_range = lidar_range
        self.set_difficulty(difficulty)
        self.rng = np.random.default_rng(0)
        self.obstacles: list[Obstacle] = []

    # ------------------------------------------------------------------ setup
    def set_difficulty(self, d: float):
        self.difficulty = float(d)
        self.dp = DifficultyParams.from_level(d)

    def _sample_start_goal(self, dist: float):
        W, m = self.WORLD, 4.0
        # Sample the direction first, then place the start inside the region from
        # which a goal `dist` away along that direction still lands in the world.
        # (Resampling start and angle independently on every retry quietly biases
        # the distance downwards at high difficulty.)
        for _ in range(200):
            phi = self.rng.uniform(-np.pi, np.pi)
            dx, dy = dist * math.cos(phi), dist * math.sin(phi)
            lo_x, hi_x = max(m, m - dx), min(W - m, W - m - dx)
            lo_y, hi_y = max(m, m - dy), min(W - m, W - m - dy)
            if lo_x < hi_x and lo_y < hi_y:
                sx = self.rng.uniform(lo_x, hi_x)
                sy = self.rng.uniform(lo_y, hi_y)
                return np.array([sx, sy]), np.array([sx + dx, sy + dy]), phi
        raise RuntimeError(f"could not place start/goal at distance {dist}")

    def _place_obstacles(self):
        obs: list[Obstacle] = []
        W = self.WORLD
        seg_a, seg_b = self.start, self.goal

        def ok(x, y, r):
            if np.hypot(x - seg_a[0], y - seg_a[1]) < r + 5.0:
                return False
            if np.hypot(x - seg_b[0], y - seg_b[1]) < r + 4.0:
                return False
            for o in obs:
                if np.hypot(x - o.x, y - o.y) < r + o.r + 1.0:
                    return False
            return True

        # Half of the static obstacles are biased towards the straight line between
        # start and goal, otherwise most episodes are trivially solved by driving
        # straight at the goal.
        for i in range(self.dp.n_static):
            for _ in range(100):
                r = self.rng.uniform(1.5, 3.5)
                if i % 2 == 0:
                    t = self.rng.uniform(0.2, 0.85)
                    p = seg_a + t * (seg_b - seg_a) + self.rng.normal(0, 4.0, 2)
                    x, y = p
                else:
                    x, y = self.rng.uniform(r, W - r, 2)
                if r < x < W - r and r < y < W - r and ok(x, y, r):
                    obs.append(Obstacle(float(x), float(y), float(r)))
                    break

        for _ in range(self.dp.n_dynamic):
            for _ in range(100):
                r = self.rng.uniform(1.2, 2.0)
                x, y = self.rng.uniform(r + 2, W - r - 2, 2)
                if ok(x, y, r + 3.0):
                    sp = self.rng.uniform(0.8, 2.0)
                    a = self.rng.uniform(-np.pi, np.pi)
                    obs.append(Obstacle(float(x), float(y), float(r),
                                        sp * math.cos(a), sp * math.sin(a)))
                    break
        self.obstacles = obs
        self._refresh_obstacle_arrays()

    def _refresh_obstacle_arrays(self):
        if self.obstacles:
            self.obs_xy = np.array([[o.x, o.y] for o in self.obstacles])
            self.obs_r = np.array([o.r for o in self.obstacles])
            self.obs_dyn = np.array([o.dynamic for o in self.obstacles])
        else:
            self.obs_xy = np.zeros((0, 2))
            self.obs_r = np.zeros(0)
            self.obs_dyn = np.zeros(0, dtype=bool)

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.start, self.goal, phi = self._sample_start_goal(self.dp.goal_dist)
        self._place_obstacles()
        # initial heading roughly towards the goal, but not exactly
        theta = float(wrap_angle(phi + self.rng.uniform(-0.6, 0.6)))
        self.state = np.array([self.start[0], self.start[1], theta, 0.0])
        self.t = 0
        self.idle_steps = 0
        self.path_length = 0.0
        self.prev_dist = float(np.linalg.norm(self.goal - self.start))
        self.done = False
        self.sensors = SensorSuite(self, self.noise,
                                   np.random.default_rng(self.rng.integers(1 << 31)))
        self.sensors.reset(self.state)
        return self.sensors.read(np.zeros(2))

    # ------------------------------------------------------------------ dynamics
    def _move_obstacles(self):
        W, dt = self.WORLD, self.vp.dt
        for o in self.obstacles:
            if not o.dynamic:
                continue
            o.x += o.vx * dt
            o.y += o.vy * dt
            if o.x < o.r or o.x > W - o.r:
                o.vx = -o.vx
                o.x = float(np.clip(o.x, o.r, W - o.r))
            if o.y < o.r or o.y > W - o.r:
                o.vy = -o.vy
                o.y = float(np.clip(o.y, o.r, W - o.r))
        self._refresh_obstacle_arrays()

    def action_to_controls(self, action):
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        return a[0] * self.vp.max_accel, a[1] * self.vp.max_steer

    def clearance(self, pos=None):
        """Distance from the vehicle edge to the nearest obstacle edge (true state)."""
        p = self.state[:2] if pos is None else pos
        if len(self.obs_r) == 0:
            return float("inf")
        d = np.linalg.norm(self.obs_xy - p, axis=1) - self.obs_r - self.vp.radius
        return float(d.min())

    def step(self, action):
        assert not self.done, "call reset() first"
        accel, steer = self.action_to_controls(action)
        x, y, th, v = self.state
        vp = self.vp
        v = float(np.clip(v + accel * vp.dt, 0.0, vp.max_speed))
        th = float(wrap_angle(th + v / vp.wheelbase * math.tan(steer) * vp.dt))
        nx, ny = x + v * math.cos(th) * vp.dt, y + v * math.sin(th) * vp.dt
        self.path_length += math.hypot(nx - x, ny - y)
        self.state = np.array([nx, ny, th, v])
        self._move_obstacles()
        self.t += 1

        rc = self.rc
        info = StepInfo(path_length=self.path_length)
        dist = float(np.linalg.norm(self.goal - self.state[:2]))
        r = rc.progress * (self.prev_dist - dist) - rc.time_cost
        self.prev_dist = dist

        clear = self.clearance()
        info.min_clearance = clear
        if clear < rc.prox_radius:
            r -= min(rc.prox_max, rc.prox_max * (rc.prox_radius - clear) / rc.prox_radius)

        if v < rc.idle_speed:
            self.idle_steps += 1
            r -= min(rc.idle_cap, rc.idle_base * self.idle_steps)
        else:
            self.idle_steps = 0

        W = self.WORLD
        out = not (vp.radius <= nx <= W - vp.radius and vp.radius <= ny <= W - vp.radius)
        if clear <= 0.0 or out:
            info.event = "collision"
            if out:
                info.collided_with = "wall"
            else:
                d = np.linalg.norm(self.obs_xy - self.state[:2], axis=1) - self.obs_r
                info.collided_with = "dynamic" if self.obs_dyn[int(np.argmin(d))] else "static"
            r -= rc.collision_penalty
            self.done = True
        elif dist < self.GOAL_RADIUS:
            info.event = "goal"
            r += rc.goal_bonus
            self.done = True
        elif self.t >= self.dp.max_steps:
            info.event = "timeout"
            self.done = True

        obs = self.sensors.read(np.array([accel, steer]))
        return obs, float(r), self.done, info

"""
Policy observation, built only from sensor data (EKF estimate + LiDAR) and the
goal position. Nothing here touches env.state.

Layout (78 dims for 36 beams):
    [0]      distance to goal / world size
    [1:3]    sin, cos of goal bearing relative to the *estimated* heading
    [3]      estimated speed / max speed
    [4:6]    previous action
    [6:42]   LiDAR ranges / max range
    [42:78]  LiDAR change since last step (rough motion cue for moving obstacles)

The relative bearing is given explicitly. With only (sin, cos) of heading plus a
goal (dx, dy) the network has to learn an atan2 difference by itself, which in
practice made steering much harder to fit during behaviour cloning.
"""
from __future__ import annotations

import math

import numpy as np

from simulation.env_core import NavEnv, wrap_angle


class ObservationBuilder:
    def __init__(self, env: NavEnv):
        self.env = env
        self.dim = 6 + 2 * env.n_beams
        self.reset()

    def reset(self):
        self.prev_lidar = None
        self.prev_action = np.zeros(2)

    def __call__(self, meas, prev_action=None) -> np.ndarray:
        env = self.env
        if prev_action is not None:
            self.prev_action = np.clip(np.asarray(prev_action, dtype=float), -1, 1)
        x, y, th, v = meas.est
        gx, gy = env.goal
        dist = math.hypot(gx - x, gy - y)
        rel = float(wrap_angle(math.atan2(gy - y, gx - x) - th))
        lid = meas.lidar / env.lidar_range
        dl = np.zeros_like(lid) if self.prev_lidar is None else (lid - self.prev_lidar) * 5.0
        self.prev_lidar = lid
        head = np.array([dist / env.WORLD, math.sin(rel), math.cos(rel),
                         v / env.vp.max_speed, *self.prev_action])
        return np.concatenate([head, lid, np.clip(dl, -1, 1)]).astype(np.float32)

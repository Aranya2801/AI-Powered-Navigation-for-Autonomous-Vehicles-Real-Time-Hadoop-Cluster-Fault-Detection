"""
Simulated LiDAR / GPS / IMU and an Extended Kalman Filter that fuses GPS + IMU
into a pose estimate.

EKF state: [x, y, theta, v]
    predict: kinematic bicycle model driven by the controls actually applied
    update : GPS (x, y), IMU heading (theta), IMU/wheel speed (v)

LiDAR is a body-frame sensor, so it is ray-cast from the true pose; it does not
go through the filter. The policy gets LiDAR ranges + the EKF estimate, never the
true pose.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


@dataclass
class SensorNoise:
    lidar_sigma: float = 0.10
    lidar_dropout: float = 0.03       # dropped beams return max range
    gps_sigma: float = 0.6
    gps_bias_walk: float = 0.02       # per-step random walk on the GPS bias
    gps_bias_max: float = 1.0
    gps_degraded_p: float = 0.02      # probability a fix is "degraded"
    gps_degraded_sigma: float = 4.0
    imu_heading_sigma: float = 0.03
    imu_heading_bias: float = 0.02    # fixed per-episode bias, sampled +-
    imu_speed_sigma: float = 0.08

    def scaled(self, k: float) -> "SensorNoise":
        """All noise magnitudes multiplied by k (probabilities are not scaled)."""
        return replace(
            self,
            lidar_sigma=self.lidar_sigma * k,
            gps_sigma=self.gps_sigma * k,
            gps_bias_walk=self.gps_bias_walk * k,
            gps_bias_max=self.gps_bias_max * k,
            gps_degraded_sigma=self.gps_degraded_sigma * k,
            imu_heading_sigma=self.imu_heading_sigma * k,
            imu_heading_bias=self.imu_heading_bias * k,
            imu_speed_sigma=self.imu_speed_sigma * k,
        )


@dataclass
class Measurement:
    lidar: np.ndarray            # (n_beams,) ranges, body frame
    gps: np.ndarray              # (2,)
    gps_degraded: bool
    imu_heading: float
    imu_speed: float
    est: np.ndarray              # (4,) EKF estimate [x, y, theta, v]
    est_cov: np.ndarray          # (4, 4)


def lidar_scan(pos, heading, obs_xy, obs_r, n_beams, max_range, world):
    """Vectorised ray cast against circles and the four walls."""
    ang = heading + np.arange(n_beams) * (2 * np.pi / n_beams)
    u = np.stack([np.cos(ang), np.sin(ang)], axis=1)            # (B, 2)
    ranges = np.full(n_beams, max_range)

    if len(obs_r):
        c = obs_xy - pos                                          # (N, 2)
        b = u @ c.T                                               # (B, N)
        cc = np.sum(c * c, axis=1) - obs_r ** 2                   # (N,)
        disc = b ** 2 - cc[None, :]
        with np.errstate(invalid="ignore"):
            t = b - np.sqrt(disc)
        # inside a circle (cc < 0) the hit is at distance 0
        t = np.where(cc[None, :] < 0, 0.0, t)
        t = np.where((disc >= 0) & (t >= 0), t, np.inf)
        ranges = np.minimum(ranges, t.min(axis=1))

    with np.errstate(divide="ignore", invalid="ignore"):
        tx = np.where(u[:, 0] > 0, (world - pos[0]) / u[:, 0],
                      np.where(u[:, 0] < 0, -pos[0] / u[:, 0], np.inf))
        ty = np.where(u[:, 1] > 0, (world - pos[1]) / u[:, 1],
                      np.where(u[:, 1] < 0, -pos[1] / u[:, 1], np.inf))
    ranges = np.minimum(ranges, np.minimum(tx, ty))
    return np.clip(ranges, 0.0, max_range)


class EKF:
    def __init__(self, wheelbase: float, dt: float, max_speed: float):
        self.L, self.dt, self.vmax = wheelbase, dt, max_speed
        self.Q = np.diag([0.02, 0.02, 0.003, 0.05]) ** 2 * 10
        self.x = np.zeros(4)
        self.P = np.eye(4)
        self.gps_rejected = 0

    def reset(self, x0, P0=None):
        self.x = np.array(x0, dtype=float)
        self.P = np.diag([1.0, 1.0, 0.1, 0.1]) if P0 is None else P0
        self.gps_rejected = 0

    def predict(self, accel, steer):
        x, y, th, v = self.x
        dt = self.dt
        k = math.tan(steer) / self.L
        vn = float(np.clip(v + accel * dt, 0.0, self.vmax))
        thn = th + vn * k * dt
        c, s = math.cos(thn), math.sin(thn)
        self.x = np.array([x + vn * c * dt, y + vn * s * dt, _wrap(thn), vn])

        F = np.eye(4)
        F[2, 3] = k * dt
        F[0, 2] = -vn * s * dt
        F[1, 2] = vn * c * dt
        F[0, 3] = c * dt - vn * s * dt * k * dt
        F[1, 3] = s * dt + vn * c * dt * k * dt
        self.P = F @ self.P @ F.T + self.Q

    def _update(self, z, H, R, angle_idx=None, gate=None):
        y = z - H @ self.x
        if angle_idx is not None:
            # without this the innovation jumps by ~2pi when heading crosses +-pi
            y[angle_idx] = _wrap(y[angle_idx])
        S = H @ self.P @ H.T + R
        Si = np.linalg.inv(S)
        if gate is not None and float(y @ Si @ y) > gate:
            return False
        K = self.P @ H.T @ Si
        self.x = self.x + K @ y
        self.x[2] = _wrap(self.x[2])
        I_KH = np.eye(4) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T      # Joseph form
        return True

    def update_gps(self, z, sigma):
        H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
        # chi2(2) 99.9% gate, catches degraded fixes the receiver didn't flag
        if not self._update(z, H, np.eye(2) * sigma ** 2, gate=13.8):
            self.gps_rejected += 1

    def update_imu(self, heading, speed, h_sigma, v_sigma):
        H = np.array([[0, 0, 1.0, 0], [0, 0, 0, 1.0]])
        R = np.diag([h_sigma ** 2, v_sigma ** 2])
        self._update(np.array([heading, speed]), H, R, angle_idx=0)


class SensorSuite:
    def __init__(self, env, noise: SensorNoise, rng: np.random.Generator):
        self.env, self.n, self.rng = env, noise, rng
        vp = env.vp
        self.ekf = EKF(vp.wheelbase, vp.dt, vp.max_speed)

    def reset(self, true_state):
        n = self.n
        self.gps_bias = np.zeros(2)
        self.heading_bias = self.rng.uniform(-n.imu_heading_bias, n.imu_heading_bias)
        # start estimate is perturbed so the filter has something to converge from
        x0 = true_state + np.array([self.rng.normal(0, n.gps_sigma),
                                    self.rng.normal(0, n.gps_sigma),
                                    self.rng.normal(0, 0.05), 0.0])
        x0[2] = _wrap(x0[2])
        self.ekf.reset(x0)

    def read(self, controls) -> Measurement:
        env, n, rng = self.env, self.n, self.rng
        x, y, th, v = env.state

        self.ekf.predict(float(controls[0]), float(controls[1]))

        # GPS with slowly drifting bias and occasional degraded fixes
        self.gps_bias = np.clip(self.gps_bias + rng.normal(0, n.gps_bias_walk, 2),
                                -n.gps_bias_max, n.gps_bias_max)
        degraded = rng.random() < n.gps_degraded_p
        sig = n.gps_degraded_sigma if degraded else n.gps_sigma
        gps = np.array([x, y]) + self.gps_bias + rng.normal(0, sig, 2)
        # the receiver reports fix quality, so the filter is told about degradation
        self.ekf.update_gps(gps, sig + n.gps_bias_max * 0.5)

        hd = float(_wrap(th + self.heading_bias + rng.normal(0, n.imu_heading_sigma)))
        sp = max(0.0, v + rng.normal(0, n.imu_speed_sigma))
        self.ekf.update_imu(hd, sp, n.imu_heading_sigma + n.imu_heading_bias,
                            n.imu_speed_sigma)

        lidar = lidar_scan(np.array([x, y]), th, env.obs_xy, env.obs_r,
                           env.n_beams, env.lidar_range, env.WORLD)
        if n.lidar_sigma > 0:
            lidar = lidar + rng.normal(0, n.lidar_sigma, lidar.shape)
        drop = rng.random(lidar.shape) < n.lidar_dropout
        lidar = np.where(drop, env.lidar_range, np.clip(lidar, 0.0, env.lidar_range))

        return Measurement(lidar, gps, degraded, hd, sp,
                           self.ekf.x.copy(), self.ekf.P.copy())

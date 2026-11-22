"""
Classical baselines: grid A*, RRT, a pure-pursuit tracker, and A* with periodic
replanning.

Information given to the baselines:
  * the true obstacle map (positions + radii at planning time) - standard for
    planning baselines, and strictly more than the RL policy gets
  * the EKF pose estimate for tracking - the same localisation the policy uses,
    so differences come from how obstacles are perceived, not from localisation

Dynamic obstacles are only known as a snapshot at planning time. Without
replanning A* is blind to where they move afterwards.
"""
from __future__ import annotations

import heapq
import math
import time

import numpy as np

from simulation.env_core import NavEnv, wrap_angle


# ---------------------------------------------------------------- geometry
def segment_clear(p, q, centers, radii):
    """True if the segment p->q stays outside every (already inflated) circle."""
    if len(radii) == 0:
        return True
    d = q - p
    L2 = float(d @ d)
    if L2 < 1e-12:
        return bool(np.all(np.linalg.norm(centers - p, axis=1) > radii))
    t = np.clip(((centers - p) @ d) / L2, 0.0, 1.0)
    closest = p + t[:, None] * d
    return bool(np.all(np.linalg.norm(centers - closest, axis=1) > radii))


def shortcut(path, centers, radii):
    """Greedy line-of-sight smoothing (string pulling)."""
    if len(path) <= 2:
        return path
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not segment_clear(path[i], path[j], centers, radii):
            j -= 1
        out.append(path[j])
        i = j
    return np.array(out)


def obstacle_snapshot(env: NavEnv, margin: float, dyn_margin: float):
    r = env.obs_r + env.vp.radius + np.where(env.obs_dyn, dyn_margin, margin)
    return env.obs_xy.copy(), r


# ---------------------------------------------------------------- A*
class GridAStar:
    def __init__(self, world: float, res: float = 0.5):
        self.res = res
        self.n = int(math.ceil(world / res))
        g = (np.arange(self.n) + 0.5) * res
        self.gx, self.gy = np.meshgrid(g, g, indexing="ij")
        self.world = world

    def occupancy(self, centers, radii, wall_margin):
        occ = np.zeros((self.n, self.n), dtype=bool)
        for (cx, cy), r in zip(centers, radii):
            occ |= (self.gx - cx) ** 2 + (self.gy - cy) ** 2 <= r * r
        w = wall_margin
        occ |= (self.gx < w) | (self.gx > self.world - w)
        occ |= (self.gy < w) | (self.gy > self.world - w)
        return occ

    def to_cell(self, p):
        i = int(np.clip(p[0] / self.res, 0, self.n - 1))
        j = int(np.clip(p[1] / self.res, 0, self.n - 1))
        return i, j

    def plan(self, start, goal, occ, max_expansions=60000):
        n = self.n
        s, g = self.to_cell(start), self.to_cell(goal)
        occ = occ.copy()
        # the vehicle may legitimately sit inside an inflation halo; free a small
        # patch around start/goal so the search can get out of it
        for c in (s, g):
            occ[max(0, c[0] - 2):c[0] + 3, max(0, c[1] - 2):c[1] + 3] = False

        SQ2 = math.sqrt(2)
        moves = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                 (1, 1, SQ2), (1, -1, SQ2), (-1, 1, SQ2), (-1, -1, SQ2)]

        def h(c):
            dx, dy = abs(c[0] - g[0]), abs(c[1] - g[1])
            return (dx + dy) + (SQ2 - 2) * min(dx, dy)

        gcost = np.full((n, n), np.inf)
        parent = {}
        gcost[s] = 0.0
        heap = [(h(s), 0.0, s)]
        exp = 0
        while heap:
            f, gc, c = heapq.heappop(heap)
            if c == g:
                break
            if gc > gcost[c]:
                continue
            exp += 1
            if exp > max_expansions:
                return None
            for dx, dy, w in moves:
                nb = (c[0] + dx, c[1] + dy)
                if not (0 <= nb[0] < n and 0 <= nb[1] < n) or occ[nb]:
                    continue
                if dx and dy and (occ[c[0] + dx, c[1]] or occ[c[0], c[1] + dy]):
                    continue  # no squeezing diagonally between two blocked cells
                ng = gc + w
                if ng < gcost[nb]:
                    gcost[nb] = ng
                    parent[nb] = c
                    heapq.heappush(heap, (ng + h(nb), ng, nb))
        else:
            return None
        if g not in parent and g != s:
            return None
        cells = [g]
        while cells[-1] != s:
            cells.append(parent[cells[-1]])
        cells.reverse()
        pts = (np.array(cells, dtype=float) + 0.5) * self.res
        pts[0], pts[-1] = start, goal
        return pts


# ---------------------------------------------------------------- RRT
class RRT:
    def __init__(self, world, step=1.5, goal_bias=0.1, max_iters=6000, rng=None):
        self.world, self.step, self.goal_bias, self.max_iters = world, step, goal_bias, max_iters
        self.rng = rng or np.random.default_rng(0)

    def plan(self, start, goal, centers, radii, wall_margin):
        W, m = self.world, wall_margin
        nodes = np.zeros((self.max_iters + 1, 2))
        parent = np.full(self.max_iters + 1, -1)
        nodes[0] = start
        k = 1
        for _ in range(self.max_iters):
            target = goal if self.rng.random() < self.goal_bias else self.rng.uniform(m, W - m, 2)
            d = np.linalg.norm(nodes[:k] - target, axis=1)
            i = int(np.argmin(d))
            vec = target - nodes[i]
            L = float(np.linalg.norm(vec))
            if L < 1e-9:
                continue
            new = nodes[i] + vec / L * min(self.step, L)
            if not (m <= new[0] <= W - m and m <= new[1] <= W - m):
                continue
            if not segment_clear(nodes[i], new, centers, radii):
                continue
            nodes[k], parent[k] = new, i
            k += 1
            if np.linalg.norm(new - goal) < self.step and segment_clear(new, goal, centers, radii):
                path = [goal, new]
                j = i
                while j != -1:
                    path.append(nodes[j])
                    j = parent[j]
                return np.array(path[::-1])
        return None


# ---------------------------------------------------------------- tracking
class PurePursuit:
    def __init__(self, env: NavEnv, lookahead=2.5, cruise=3.0):
        self.vp = env.vp
        self.lookahead, self.cruise = lookahead, cruise

    def act(self, est, path, idx_hint=0):
        x, y, th, v = est
        p = np.array([x, y])
        # closest point at or after the previous index (never go backwards)
        seg = path[idx_hint:]
        i_close = idx_hint + int(np.argmin(np.linalg.norm(seg - p, axis=1)))
        target = path[-1]
        for j in range(i_close, len(path)):
            if np.linalg.norm(path[j] - p) >= self.lookahead:
                target = path[j]
                break
        # interpolate along segments so long smoothed segments still give a
        # target at ~lookahead distance
        if np.linalg.norm(target - p) > 2 * self.lookahead and i_close + 1 < len(path):
            a, b = path[i_close], path[min(i_close + 1, len(path) - 1)]
            d = b - a
            L = np.linalg.norm(d)
            if L > 1e-6:
                t = np.clip(((p - a) @ d) / (L * L), 0, 1)
                proj = a + t * d
                target = proj + d / L * self.lookahead
        alpha = float(wrap_angle(math.atan2(target[1] - y, target[0] - x) - th))
        ld = max(1e-3, float(np.linalg.norm(target - p)))
        steer = math.atan2(2 * self.vp.wheelbase * math.sin(alpha), ld)
        v_ref = self.cruise * max(0.35, math.cos(alpha))
        accel = (v_ref - v) / self.vp.dt
        return np.array([np.clip(accel / self.vp.max_accel, -1, 1),
                         np.clip(steer / self.vp.max_steer, -1, 1)]), i_close


# ---------------------------------------------------------------- agents
class PlannerAgent:
    """Plan once (or every `replan_every` steps) and track with pure pursuit."""

    name = "planner"

    def __init__(self, kind="astar", replan_every: int | None = None,
                 margin=1.0, dyn_margin=1.5, seed=0):
        self.kind, self.replan_every = kind, replan_every
        self.margin, self.dyn_margin = margin, dyn_margin
        self.seed = seed
        self.name = {("astar", None): "A*", ("rrt", None): "RRT"}.get(
            (kind, replan_every), f"{kind.upper()}+replan/{replan_every}")
        if kind == "astar" and replan_every:
            self.name = "A* (replan)"

    def reset(self, env: NavEnv, meas, episode_seed: int = 0):
        self.env = env
        self.astar = GridAStar(env.WORLD)
        self.rrt = RRT(env.WORLD, rng=np.random.default_rng(episode_seed))
        self.tracker = PurePursuit(env)
        self.path, self.idx, self.k = None, 0, 0
        self.plan_time = 0.0
        self.n_plans = 0
        self._plan(meas)

    def _plan(self, meas):
        t0 = time.perf_counter()
        env = self.env
        centers, radii = obstacle_snapshot(env, self.margin, self.dyn_margin)
        start = meas.est[:2].copy()
        wall = env.vp.radius + 0.3
        if self.kind == "astar":
            occ = self.astar.occupancy(centers, radii, wall)
            path = self.astar.plan(start, env.goal, occ)
        else:
            path = self.rrt.plan(start, env.goal, centers, radii, wall)
        if path is not None:
            self.path = shortcut(path, centers, radii)
            self.idx = 0
        elif self.replan_every is None:
            self.path = None
        self.plan_time += time.perf_counter() - t0
        self.n_plans += 1

    def act(self, meas):
        self.k += 1
        if self.replan_every and self.k % self.replan_every == 0:
            self._plan(meas)
        if self.path is None:
            return np.array([-1.0, 0.0])  # no plan: brake and wait
        a, self.idx = self.tracker.act(meas.est, self.path, self.idx)
        return a

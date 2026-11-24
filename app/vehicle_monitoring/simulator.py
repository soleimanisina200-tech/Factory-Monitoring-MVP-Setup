# app/vehicle_monitoring/simulator.py
"""
Synthetic 3-D world for measuring the driver state machine against GROUND TRUTH.

There is no labelled forklift footage of the site, and a few hand-made boxes cannot tell how often the wrong
person becomes "the driver". This simulator produces what the real pipeline feeds the state machine - vehicle boxes
/ contours and person boxes with COCO keypoints, foot point and ids - from a 3-D scene seen by a pinhole camera:

  * forklifts are oriented 3-D boxes with a chassis, mast uprights and optionally an overhead guard ("roof") or a
    closed cabin; they hide the body parts behind / inside them (ray test per keypoint), exactly the reason real
    drivers and bystanders are hard to tell apart;
  * people have a 3-D skeleton (standing / walking / climbing in / seated in the cab);
  * detections are noisy: jittered keypoints, low-confidence (sometimes wrongly confident) hidden joints, boxes of
    the visible part only, missed detections;
  * the identity layer is imitated too: provisional "T<n>" ids that become worker ids, new tracks after an
    occlusion, occasionally a NEW id for the same person, "left the camera" notifications.

Scripts (Python generators) move the actors; the ground truth (who drives which forklift, onboard or not, true
distance) is recorded every frame and compared with the state machine (see `Scores`).
It is a model, not reality: it measures the LOGIC under controlled difficulty, and makes regressions visible.
The numbers are not the accuracy on real video.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import numpy as np
import yaml

from app.logic.geometry import MetricBEVProjector
from app.vehicle_monitoring.distance_estimator import RealWorldDistanceEstimator, point_polygon_distance
from app.vehicle_monitoring.driver_vehicle_fsm import DriverVehicleMonitorFSM

IMG_W, IMG_H = 1920, 1080
Script = Generator[None, None, None]


# ============================================================================================== camera
class Camera:
    def __init__(self, pos: Tuple[float, float, float], look_at: Tuple[float, float, float], f: float = 1050.0,
                 name: str = "cam"):
        self.name = name
        self.pos = np.asarray(pos, dtype=np.float64)
        fwd = np.asarray(look_at, dtype=np.float64) - self.pos
        self.fwd = fwd / np.linalg.norm(fwd)
        right = np.cross(self.fwd, np.array([0.0, 0.0, 1.0]))
        self.right = right / np.linalg.norm(right)
        self.down = np.cross(self.fwd, self.right)
        self.f = float(f)
        # ground homography (pixel -> metres), as produced by the calibration tool
        ground = np.array([[-3, 2, 0], [3, 2, 0], [3, 8, 0], [-3, 8, 0]], dtype=np.float64)
        self.H_px_to_world = _homography(self.project(ground)[:, :2], ground[:, :2])

    def project(self, pts: np.ndarray) -> np.ndarray:
        """(N,3) world -> (N,3) [u, v, depth]."""
        d = np.atleast_2d(pts) - self.pos
        z = (d * self.fwd).sum(axis=1)
        zs = np.maximum(z, 1e-6)
        u = IMG_W / 2.0 + self.f * (d * self.right).sum(axis=1) / zs
        v = IMG_H / 2.0 + self.f * (d * self.down).sum(axis=1) / zs
        return np.column_stack([u, v, z])


def _homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """3x3 homography src -> dst from 4 point pairs (direct linear solution; no OpenCV in the simulation loop)."""
    a, b = [], []
    for (x, y), (u, v) in zip(np.asarray(src, dtype=np.float64), np.asarray(dst, dtype=np.float64)):
        a.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        a.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        b += [u, v]
    h = np.linalg.solve(np.asarray(a), np.asarray(b))
    return np.append(h, 1.0).reshape(3, 3)


CAMERAS = {
    "oblique": lambda: Camera((0.0, -9.0, 6.0), (0.0, 3.5, 0.0), name="oblique"),
    "side_low": lambda: Camera((-9.0, -5.0, 4.5), (0.5, 4.0, 0.0), name="side_low"),
    "steep": lambda: Camera((0.0, -3.5, 9.0), (0.0, 4.0, 0.0), f=900.0, name="steep"),
}


# ============================================================================================== geometry
def _hidden_by_box(origin: np.ndarray, pts: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """True where the segment origin -> point passes through the axis-aligned box [lo, hi] before the point."""
    d = pts - origin
    n = len(pts)
    t0 = np.zeros(n)
    t1 = np.full(n, 1.0 - 2e-3)
    ok = np.ones(n, dtype=bool)
    for i in range(3):
        di = d[:, i]
        par = np.abs(di) < 1e-9
        if origin[i] < lo[i] or origin[i] > hi[i]:
            ok &= ~par
        safe = np.where(par, 1.0, di)
        ta, tb = (lo[i] - origin[i]) / safe, (hi[i] - origin[i]) / safe
        lo_t = np.where(par, -np.inf, np.minimum(ta, tb))
        hi_t = np.where(par, np.inf, np.maximum(ta, tb))
        t0, t1 = np.maximum(t0, lo_t), np.minimum(t1, hi_t)
    return ok & (t0 <= t1)


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Monotone chain (the outline of the projected vehicle box = what a segmentation mask would show)."""
    pts = sorted(map(tuple, np.asarray(points, dtype=np.float64)))
    if len(pts) < 3:
        return np.asarray(pts)

    def half(seq):
        out: List[Tuple[float, float]] = []
        for p in seq:
            while len(out) >= 2 and ((out[-1][0] - out[-2][0]) * (p[1] - out[-2][1])
                                     - (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0])) <= 0:
                out.pop()
            out.append(p)
        return out

    lower, upper = half(pts), half(reversed(pts))
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


# ============================================================================================== actors
@dataclass
class SimVehicle:
    vid: int
    x: float
    y: float
    heading: float = 0.0
    kind: str = "open"                 # open | roof (overhead guard) | cabin (closed: the driver is never seen)
    driver: Optional["SimPerson"] = None
    speed: float = 0.0

    HALF_L, HALF_W, TOP = 1.2, 0.6, 2.2
    SEAT = (-0.25, 0.0)

    def to_world(self, local: np.ndarray) -> np.ndarray:
        local = np.atleast_2d(np.asarray(local, dtype=np.float64))
        c, s = math.cos(self.heading), math.sin(self.heading)
        out = local.copy()
        out[:, 0] = self.x + c * local[:, 0] - s * local[:, 1]
        out[:, 1] = self.y + s * local[:, 0] + c * local[:, 1]
        return out

    def to_local(self, world: np.ndarray) -> np.ndarray:
        world = np.atleast_2d(np.asarray(world, dtype=np.float64))
        c, s = math.cos(self.heading), math.sin(self.heading)
        dx, dy = world[:, 0] - self.x, world[:, 1] - self.y
        out = world.copy()
        out[:, 0] = c * dx + s * dy
        out[:, 1] = -s * dx + c * dy
        return out

    def occluders(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Solid parts in the vehicle frame (x forward, y left, z up)."""
        b = [((-1.2, -0.6, 0.0), (1.2, 0.6, 0.95)),                 # chassis / counterweight / battery
             ((1.2, -0.50, 0.0), (1.35, -0.34, 2.2)),               # mast uprights
             ((1.2, 0.34, 0.0), (1.35, 0.50, 2.2))]
        if self.kind in ("roof", "cabin"):
            b.append(((-0.95, -0.6, 2.08), (0.55, 0.6, 2.18)))      # overhead guard
        if self.kind == "cabin":                                     # closed cab: panels / glass with glare
            b += [((-0.95, -0.6, 0.95), (-0.90, 0.6, 2.08)), ((0.50, -0.6, 0.95), (0.55, 0.6, 2.08)),
                  ((-0.95, -0.6, 0.95), (0.55, -0.55, 2.08)), ((-0.95, 0.55, 0.95), (0.55, 0.6, 2.08))]
        return [(np.array(lo), np.array(hi)) for lo, hi in b]

    def hides(self, cam: Camera, pts: np.ndarray) -> np.ndarray:
        origin = self.to_local(cam.pos[None, :])[0]
        local = self.to_local(pts)
        hidden = np.zeros(len(pts), dtype=bool)
        for lo, hi in self.occluders():
            hidden |= _hidden_by_box(origin, local, lo, hi)
        return hidden

    def outline_points(self) -> np.ndarray:
        pts = [(sx * 1.2 if sx < 0 else 1.35, sy * 0.6, z) for sx in (-1, 1) for sy in (-1, 1) for z in (0.0, self.TOP)]
        pts += [(2.2, -0.45, 0.0), (2.2, 0.45, 0.0), (2.2, -0.45, 0.1), (2.2, 0.45, 0.1)]     # forks
        return self.to_world(np.array(pts))

    def footprint(self) -> np.ndarray:
        return self.to_world(np.array([(-1.2, -0.6, 0), (2.2, -0.6, 0), (2.2, 0.6, 0), (-1.2, 0.6, 0)]))[:, :2]

    def blocks(self, a: Tuple[float, float], b: Tuple[float, float], margin: float = 0.3) -> bool:
        """Does the straight walk a -> b pass through the body of the vehicle?"""
        la, lb = self.to_local(np.array([[a[0], a[1], 0.0], [b[0], b[1], 0.0]]))[:, :2]
        lo, hi = np.array([-1.2 - margin, -0.6 - margin]), np.array([1.35 + margin, 0.6 + margin])
        d = lb - la
        t0, t1 = 0.0, 1.0
        for i in range(2):
            if abs(d[i]) < 1e-9:
                if la[i] < lo[i] or la[i] > hi[i]:
                    return False
            else:
                ta, tb = (lo[i] - la[i]) / d[i], (hi[i] - la[i]) / d[i]
                t0, t1 = max(t0, min(ta, tb)), min(t1, max(ta, tb))
                if t0 > t1:
                    return False
        return True

    def detour(self, a: Tuple[float, float], b: Tuple[float, float]) -> List[Tuple[float, float]]:
        """Way points around the vehicle for the walk a -> b (empty when the straight line is free)."""
        if not self.blocks(a, b):
            return []
        m = 0.75
        corners = self.to_world(np.array([(-1.2 - m, -0.6 - m, 0), (1.35 + m, -0.6 - m, 0),
                                          (1.35 + m, 0.6 + m, 0), (-1.2 - m, 0.6 + m, 0)]))[:, :2]
        best, best_len = [], float("inf")
        for i in range(4):
            c = (float(corners[i][0]), float(corners[i][1]))
            routes = [[c]]
            for j in ((i + 1) % 4, (i - 1) % 4):
                routes.append([c, (float(corners[j][0]), float(corners[j][1]))])
            for r in routes:
                pts = [a] + r + [b]
                if any(self.blocks(pts[k], pts[k + 1]) for k in range(len(pts) - 1)):
                    continue
                length = sum(math.hypot(pts[k + 1][0] - pts[k][0], pts[k + 1][1] - pts[k][1])
                             for k in range(len(pts) - 1))
                if length < best_len:
                    best, best_len = r, length
        return best

    # ---- script helpers
    def wait(self, sim: "DriverSim", sec: float) -> Script:
        self.speed = 0.0
        for _ in range(max(1, int(round(sec / sim.dt)))):
            yield

    def wait_for_driver(self, sim: "DriverSim", extra: float = 1.5) -> Script:
        while self.driver is None or not self.driver.seated:
            yield
        yield from self.wait(sim, extra)

    def drive_to(self, sim: "DriverSim", x: float, y: float, speed: float = 2.0) -> Script:
        while True:
            dx, dy = x - self.x, y - self.y
            dist = math.hypot(dx, dy)
            if dist < speed * sim.dt:
                self.x, self.y, self.speed = x, y, 0.0
                return
            want = math.atan2(dy, dx)
            diff = (want - self.heading + math.pi) % (2 * math.pi) - math.pi
            turn = math.radians(70.0) * sim.dt
            if abs(diff) > turn:                      # turn (slowly creeping) before driving straight
                self.heading += math.copysign(turn, diff)
                v = 0.4
            else:
                self.heading = want
                v = speed
            self.speed = v
            self.x += math.cos(self.heading) * v * sim.dt
            self.y += math.sin(self.heading) * v * sim.dt
            yield


_STAND = {  # forward, left, up (metres, 1.72 m person)
    0: (0.10, 0.0, 1.62), 1: (0.09, 0.03, 1.65), 2: (0.09, -0.03, 1.65), 3: (0.0, 0.08, 1.63), 4: (0.0, -0.08, 1.63),
    5: (0.0, 0.20, 1.43), 6: (0.0, -0.20, 1.43), 7: (0.02, 0.24, 1.13), 8: (0.02, -0.24, 1.13),
    9: (0.06, 0.24, 0.87), 10: (0.06, -0.24, 0.87), 11: (0.0, 0.11, 0.93), 12: (0.0, -0.11, 0.93),
    13: (0.02, 0.11, 0.50), 14: (0.02, -0.11, 0.50), 15: (0.0, 0.10, 0.08), 16: (0.0, -0.10, 0.08),
    17: (0.0, 0.0, 1.74), 18: (0.05, 0.10, 0.0), 19: (0.05, -0.10, 0.0),            # head top, soles
}
_SEAT = {   # relative to the seat point of the vehicle (vehicle frame)
    0: (0.02, 0.0, 1.76), 1: (0.01, 0.03, 1.79), 2: (0.01, -0.03, 1.79), 3: (-0.08, 0.08, 1.77),
    4: (-0.08, -0.08, 1.77), 5: (-0.10, 0.20, 1.55), 6: (-0.10, -0.20, 1.55), 7: (0.10, 0.25, 1.30),
    8: (0.10, -0.25, 1.30), 9: (0.35, 0.18, 1.30), 10: (0.35, -0.18, 1.30), 11: (-0.05, 0.11, 1.05),
    12: (-0.05, -0.11, 1.05), 13: (0.38, 0.12, 1.08), 14: (0.38, -0.12, 1.08), 15: (0.50, 0.12, 0.62),
    16: (0.50, -0.12, 0.62), 17: (-0.06, 0.0, 1.88), 18: (0.55, 0.12, 0.55), 19: (0.55, -0.12, 0.55),
}
_STAND_A = np.array([_STAND[i] for i in range(20)], dtype=np.float64)
_SEAT_A = np.array([_SEAT[i] for i in range(20)], dtype=np.float64)


@dataclass
class SimPerson:
    pid: int
    x: float
    y: float
    heading: float = 0.0
    height: float = 1.72
    vehicle: Optional[SimVehicle] = None
    seated: bool = False
    climb: float = 0.0                 # 0 = standing next to the vehicle ... 1 = seated
    climbing_on: Optional[SimVehicle] = None
    moving: bool = False
    phase: float = 0.0
    gone: bool = False                 # left the scene for good

    def body_points(self) -> np.ndarray:
        """(20,3) world points: 17 COCO joints, head top, two soles."""
        s = self.height / 1.72
        stand = _STAND_A * s
        if self.moving:                                    # gait: legs swing, a foot lifts
            sw = math.sin(self.phase)
            stand = stand.copy()
            for side, (ank, knee, sole) in ((1, (15, 13, 18)), (-1, (16, 14, 19))):
                stand[ank, 0] += 0.22 * sw * side
                stand[sole, 0] += 0.22 * sw * side
                stand[knee, 0] += 0.10 * sw * side
                stand[ank, 2] += 0.07 * max(0.0, math.cos(self.phase) * side)
        c, sn = math.cos(self.heading), math.sin(self.heading)
        world = np.empty_like(stand)
        world[:, 0] = self.x + c * stand[:, 0] - sn * stand[:, 1]
        world[:, 1] = self.y + sn * stand[:, 0] + c * stand[:, 1]
        world[:, 2] = stand[:, 2]
        veh = self.vehicle if self.seated else self.climbing_on
        if veh is None:
            return world
        seat = _SEAT_A.copy()
        seat[:, 2] = 1.05 + (seat[:, 2] - 1.05) * s
        seat[:, 0] += SimVehicle.SEAT[0]
        seat[:, 1] += SimVehicle.SEAT[1]
        seated_world = veh.to_world(seat)
        if self.seated:
            return seated_world
        return (1.0 - self.climb) * world + self.climb * seated_world

    def ground_xy(self) -> Tuple[float, float]:
        if self.seated and self.vehicle is not None:
            p = self.vehicle.to_world(np.array([[SimVehicle.SEAT[0], SimVehicle.SEAT[1], 0.0]]))[0]
            return float(p[0]), float(p[1])
        return self.x, self.y

    # ---- script helpers
    def wait(self, sim: "DriverSim", sec: float) -> Script:
        self.moving = False
        for _ in range(max(1, int(round(sec / sim.dt)))):
            yield

    def walk_to(self, sim: "DriverSim", x: float, y: float, speed: float = 1.3) -> Script:
        for v in sim.vehicles:                      # people do not walk through a forklift
            if v.speed == 0.0 and self.vehicle is not v:
                for wx, wy in v.detour((self.x, self.y), (x, y)):
                    yield from self._walk_straight(sim, wx, wy, speed)
        yield from self._walk_straight(sim, x, y, speed)

    def _walk_straight(self, sim: "DriverSim", x: float, y: float, speed: float = 1.3) -> Script:
        while True:
            dx, dy = x - self.x, y - self.y
            dist = math.hypot(dx, dy)
            if dist <= speed * sim.dt:
                self.x, self.y, self.moving = x, y, False
                return
            self.heading = math.atan2(dy, dx)
            self.x += dx / dist * speed * sim.dt
            self.y += dy / dist * speed * sim.dt
            self.moving = True
            self.phase += 2 * math.pi * 1.8 * sim.dt
            yield

    def follow(self, sim: "DriverSim", veh: SimVehicle, local: Tuple[float, float], sec: float) -> Script:
        """Stay at a point given in the vehicle frame (walks along when the vehicle moves)."""
        for _ in range(max(1, int(round(sec / sim.dt)))):
            t = veh.to_world(np.array([[local[0], local[1], 0.0]]))[0]
            way = veh.detour((self.x, self.y), (float(t[0]), float(t[1]))) if veh.speed == 0.0 else []
            if way:
                t = (way[0][0], way[0][1])
            d = math.hypot(t[0] - self.x, t[1] - self.y)
            self.moving = d > 0.05
            if self.moving:
                step = min(d, 1.6 * sim.dt)
                self.heading = math.atan2(t[1] - self.y, t[0] - self.x)
                self.x += (t[0] - self.x) / d * step
                self.y += (t[1] - self.y) / d * step
                self.phase += 2 * math.pi * 1.8 * sim.dt
            yield
        self.moving = False

    def board(self, sim: "DriverSim", veh: SimVehicle, side: int = 1, climb_sec: float = 1.2) -> Script:
        t = veh.to_world(np.array([[SimVehicle.SEAT[0], side * 1.05, 0.0]]))[0]
        yield from self.walk_to(sim, float(t[0]), float(t[1]))
        self.heading = veh.heading - side * math.pi / 2.0          # face the cab
        self.climbing_on = veh
        steps = max(1, int(round(climb_sec / sim.dt)))
        for k in range(steps):
            self.climb = (k + 1) / steps
            if self.climb >= 0.5 and veh.driver is not self:
                veh.driver = self
                sim.session_driver[veh.vid] = self.pid
                for other in sim.vehicles:                         # one person drives one vehicle
                    if other is not veh and sim.session_driver.get(other.vid) == self.pid:
                        sim.session_driver[other.vid] = None
            yield
        self.vehicle, self.seated, self.climbing_on, self.climb = veh, True, None, 1.0

    def dismount(self, sim: "DriverSim", side: int = 1, climb_sec: float = 1.0) -> Script:
        veh = self.vehicle
        assert veh is not None and self.seated
        t = veh.to_world(np.array([[SimVehicle.SEAT[0], side * 1.05, 0.0]]))[0]
        self.x, self.y = float(t[0]), float(t[1])
        self.heading = veh.heading + side * math.pi / 2.0          # facing away from the cab
        self.seated, self.vehicle, self.climbing_on = False, None, veh
        steps = max(1, int(round(climb_sec / sim.dt)))
        for k in range(steps):
            self.climb = 1.0 - (k + 1) / steps
            if self.climb < 0.5 and veh.driver is self:
                veh.driver = None                                  # session driver stays (he is "its" driver)
            yield
        self.climbing_on, self.climb = None, 0.0

    def leave(self, sim: "DriverSim", x: float, y: float, speed: float = 1.4) -> Script:
        yield from self.walk_to(sim, x, y, speed)
        self.gone = True


# ============================================================================================== noise / identity
@dataclass
class Noise:
    kpt_sigma: float = 0.012           # x person height in pixels
    box_sigma: float = 0.010
    p_miss_person: float = 0.03
    p_miss_vehicle: float = 0.02
    p_hallucinate: float = 0.08        # hidden joint reported with a confidence above the usual threshold
    vehicle_sigma_px: float = 2.0
    min_visible_joints: int = 4        # fewer visible joints -> the pose detector does not find the person
    provisional_sec: Tuple[float, float] = (0.3, 0.9)   # track age before the identity layer names it
    gap_reset_sec: float = 1.5         # unseen longer -> a new track
    p_new_id_after_gap: float = 0.12   # identity layer fails to recognise the person -> NEW worker id
    p_merge_later: float = 0.5         # ... and merges the two ids a few seconds later


class IdentityLayer:
    """Imitates main.py + identity manager: 'T<n>' provisional ids, worker ids, new tracks, merges, exits."""

    def __init__(self, sim: "DriverSim"):
        self.sim = sim
        self.tracks: Dict[int, Dict[str, Any]] = {}       # true pid -> track state
        self.true_of: Dict[str, int] = {}                  # reported id -> true pid
        self._next_track = 0
        self._next_gid = 100
        self._pending_merges: List[Tuple[float, str, str]] = []

    def report(self, pid: int, ts: float) -> str:
        rng, nz, fsm = self.sim.rng, self.sim.noise, self.sim.fsm
        st = self.tracks.get(pid)
        if st is None or ts - st["last"] > nz.gap_reset_sec:
            self._next_track += 1
            prev = st["gid"] if st is not None else None
            st = self.tracks[pid] = {"track": self._next_track, "start": ts, "gid": None, "prev": prev,
                                     "named_at": ts + rng.uniform(*nz.provisional_sec)}
        st["last"] = ts
        if st["gid"] is None and ts >= st["named_at"]:
            if st["prev"] is not None and rng.random() >= nz.p_new_id_after_gap:
                gid = st["prev"]
            else:
                self._next_gid += 1
                gid = str(self._next_gid)
                if st["prev"] is not None and rng.random() < nz.p_merge_later:
                    self._pending_merges.append((ts + rng.uniform(2.0, 5.0), gid, st["prev"]))
            st["gid"] = gid
            self.true_of[gid] = pid
            fsm.rename_person(f"T{st['track']}", gid)
        rid = st["gid"] if st["gid"] is not None else f"T{st['track']}"
        self.true_of[rid] = pid
        return rid

    def step(self, ts: float) -> None:
        for item in [m for m in self._pending_merges if m[0] <= ts]:
            self._pending_merges.remove(item)
            _, new, old = item
            pid = self.true_of.get(new)
            st = self.tracks.get(pid) if pid is not None else None
            if st is not None and st["gid"] == new:
                st["gid"] = old
                self.sim.fsm.rename_person(new, old)


# ============================================================================================== scores
@dataclass
class Scores:
    t_driver: float = 0.0          # vehicle-seconds with a true (session) driver
    t_correct: float = 0.0
    t_wrong: float = 0.0           # another person is bound as the driver            <- the worst error
    t_missing: float = 0.0         # no driver / unknown
    t_empty: float = 0.0           # vehicle-seconds without any true driver
    t_false: float = 0.0           # ... but somebody is bound
    onboard_total: float = 0.0
    onboard_ok: float = 0.0
    seat_true: float = 0.0
    seat_fsm: float = 0.0
    alarms_true: int = 0
    alarms_found: int = 0
    alarm_delay: float = 0.0
    alarms_false: int = 0
    runs: int = 0

    def add(self, o: "Scores") -> "Scores":
        for k in self.__dataclass_fields__:
            setattr(self, k, getattr(self, k) + getattr(o, k))
        return self

    def summary(self) -> Dict[str, float]:
        d = max(self.t_driver, 1e-9)
        return {
            "driver_correct": self.t_correct / d,
            "driver_wrong": self.t_wrong / d,
            "driver_missing": self.t_missing / d,
            "false_driver": self.t_false / max(self.t_empty, 1e-9) if self.t_empty > 0 else 0.0,
            "onboard_state": self.onboard_ok / max(self.onboard_total, 1e-9) if self.onboard_total > 0 else 1.0,
            "seat_time_error": abs(self.seat_fsm - self.seat_true) / max(self.seat_true, 1e-9)
            if self.seat_true > 0 else 0.0,
            "alarm_recall": self.alarms_found / self.alarms_true if self.alarms_true else 1.0,
            "alarm_delay_sec": self.alarm_delay / self.alarms_found if self.alarms_found else 0.0,
            "false_alarms": float(self.alarms_false),
            "alarms_true": float(self.alarms_true),
            "runs": float(self.runs),
        }


def load_fsm_config(path: str = "configs/vehicle_monitoring.yaml") -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        p = Path(__file__).resolve().parents[2] / path
    with open(p, "r", encoding="utf-8") as f:
        return (yaml.safe_load(f) or {}).get("vehicle_monitoring", {}) or {}


# ============================================================================================== the simulation
class DriverSim:
    def __init__(self, camera: Camera, seed: int = 0, fps: float = 10.0, noise: Optional[Noise] = None,
                 config: Optional[Dict[str, Any]] = None, use_contour: bool = True):
        self.cam = camera
        self.rng = np.random.default_rng(seed)
        self.dt = 1.0 / fps
        self.noise = noise or Noise()
        self.use_contour = use_contour
        cfg = dict(config if config is not None else load_fsm_config())
        cfg["calibration"] = {}
        self.estimator = RealWorldDistanceEstimator(cfg)
        self.estimator.set_homography(camera.H_px_to_world)
        self.fsm = DriverVehicleMonitorFSM(cfg, self.estimator)
        self.safe_m = float((cfg.get("distance") or {}).get("safe_distance_threshold_m", 3.0))
        self.vehicles: List[SimVehicle] = []
        self.people: List[SimPerson] = []
        self._scripts: List[Script] = []
        self.identity = IdentityLayer(self)
        self.session_driver: Dict[int, Optional[int]] = {}
        self.t = 0.0
        self.frames: List[Dict[str, Any]] = []            # per-frame ground truth + state machine output
        self.events: List[Dict[str, Any]] = []
        self._vid_alias: Dict[int, int] = {}
        self._out_of_view_since: Dict[int, float] = {}
        self._left_reported: set = set()
        self._closed = False

    # ---- building the scene
    def vehicle(self, x: float, y: float, heading_deg: float = 0.0, kind: str = "open") -> SimVehicle:
        v = SimVehicle(len(self.vehicles) + 1, x, y, math.radians(heading_deg), kind)
        self.vehicles.append(v)
        self.session_driver[v.vid] = None
        return v

    def person(self, x: float, y: float, height: float = 1.72) -> SimPerson:
        p = SimPerson(len(self.people) + 1, x, y, height=height)
        self.people.append(p)
        return p

    def script(self, gen: Script) -> None:
        self._scripts.append(gen)

    # ---- observation
    def _observe_person(self, p: SimPerson) -> Optional[Dict[str, Any]]:
        nz, rng = self.noise, self.rng
        pts = p.body_points()
        uvz = self.cam.project(pts)
        vis = (uvz[:, 2] > 0.3) & (uvz[:, 0] >= 0) & (uvz[:, 0] < IMG_W) & (uvz[:, 1] >= 0) & (uvz[:, 1] < IMG_H)
        for v in self.vehicles:
            vis &= ~v.hides(self.cam, pts)
        if int(vis[:17].sum()) < nz.min_visible_joints or rng.random() < nz.p_miss_person:
            return None
        gx, gy = p.ground_xy()                             # pixel height of this person standing here
        full_h = max(20.0, float(np.ptp(self.cam.project(np.array([[gx, gy, 0.0], [gx, gy, p.height]]))[:, 1])))
        sig = nz.kpt_sigma * full_h
        k = np.zeros((17, 3), dtype=np.float32)
        for i in range(17):
            if vis[i]:
                k[i, :2] = uvz[i, :2] + rng.normal(0, sig, 2)
                k[i, 2] = rng.uniform(0.70, 0.95)
            else:
                k[i, :2] = uvz[i, :2] + rng.normal(0, 3.0 * sig, 2)
                k[i, 2] = rng.uniform(0.32, 0.50) if rng.random() < nz.p_hallucinate else rng.uniform(0.05, 0.25)
        seen = uvz[vis]
        pad = 0.05 * full_h
        box = np.array([seen[:, 0].min() - pad, seen[:, 1].min() - 0.3 * pad, seen[:, 0].max() + pad,
                        seen[:, 1].max() + 0.3 * pad]) + rng.normal(0, nz.box_sigma * full_h, 4)
        box = np.clip(box, [0, 0, 0, 0], [IMG_W - 1, IMG_H - 1, IMG_W - 1, IMG_H - 1])
        if box[2] - box[0] < 6 or box[3] - box[1] < 10:
            return None
        u, v, extrap = MetricBEVProjector.estimate_ground_contact_point(box, k[:, :2], k[:, 2])
        bev = self.estimator.pixel_to_world((u, v))
        return {"bbox": [float(b) for b in box], "keypoints": k, "foot_px": (float(u), float(v)),
                "ground_uncertain": bool(extrap), "bev_xy": (float(bev[0]), float(bev[1])), "embedding": None}

    def _observe_vehicle(self, v: SimVehicle) -> Optional[Dict[str, Any]]:
        uvz = self.cam.project(v.outline_points())
        if (uvz[:, 2] <= 0.3).any() or self.rng.random() < self.noise.p_miss_vehicle:
            return None
        px = uvz[:, :2] + self.rng.normal(0, self.noise.vehicle_sigma_px, (len(uvz), 2))
        x1, y1, x2, y2 = px[:, 0].min(), px[:, 1].min(), px[:, 0].max(), px[:, 1].max()
        if x2 < 20 or y2 < 20 or x1 > IMG_W - 20 or y1 > IMG_H - 20:
            return None
        box = [float(max(0, x1)), float(max(0, y1)), float(min(IMG_W - 1, x2)), float(min(IMG_H - 1, y2))]
        out = {"id": str(self._vid_alias.get(v.vid, v.vid)), "bbox": box}
        if self.use_contour:
            out["contour"] = _convex_hull(px)
        return out

    def _in_view(self, p: SimPerson) -> bool:
        gx, gy = p.ground_xy()
        uvz = self.cam.project(np.array([[gx, gy, 0.9 * p.height / 1.72]]))[0]
        return bool(uvz[2] > 0.3 and 0 <= uvz[0] < IMG_W and 0 <= uvz[1] < IMG_H) and not p.gone

    # ---- one frame
    def step(self) -> None:
        self.t += self.dt
        for g in list(self._scripts):
            try:
                next(g)
            except StopIteration:
                self._scripts.remove(g)
        for p in self.people:                              # seated people ride along
            if p.seated and p.vehicle is not None:
                p.x, p.y = p.ground_xy()
        self.identity.step(self.t)

        persons, left = [], []
        for p in self.people:
            if p.gone:
                obs = None
            else:
                obs = self._observe_person(p)
            if obs is not None:
                obs["id"] = self.identity.report(p.pid, self.t)
                persons.append(obs)
            if self._in_view(p):
                self._out_of_view_since.pop(p.pid, None)
                self._left_reported.discard(p.pid)
            else:
                since = self._out_of_view_since.setdefault(p.pid, self.t)
                st = self.identity.tracks.get(p.pid)
                if self.t - since >= 0.8 and p.pid not in self._left_reported and st and st["gid"]:
                    self._left_reported.add(p.pid)
                    left.append(st["gid"])
        vehicles = [o for o in (self._observe_vehicle(v) for v in self.vehicles) if o is not None]
        recs, events = self.fsm.update(vehicles, persons, (IMG_H, IMG_W, 3), now=self.t, left_view_ids=left)
        self.events += [dict(e, emitted_ts=self.t) for e in events]
        seen_v = {o["id"] for o in vehicles}

        row: Dict[str, Any] = {"t": self.t, "vehicles": {}}
        for v in self.vehicles:
            vid = str(self._vid_alias.get(v.vid, v.vid))
            rec = recs.get(vid)
            true_pid = self.session_driver.get(v.vid)
            drv = next((p for p in self.people if p.pid == true_pid), None)
            onboard = drv is not None and (v.driver is drv)
            dist = None
            if drv is not None and not onboard:
                dist = point_polygon_distance((drv.x, drv.y), v.footprint())[0]
            fsm_pid = rec.associated_driver_id if rec is not None else None
            row["vehicles"][v.vid] = {
                "seen": vid in seen_v, "true_driver": true_pid, "true_onboard": onboard, "true_dist": dist,
                "driver_in_view": drv is not None and self._in_view(drv),
                "fsm_driver": fsm_pid, "fsm_true": self.identity.true_of.get(fsm_pid) if fsm_pid else None,
                "fsm_onboard": bool(rec.onboard) if rec is not None else False,
                "fsm_alarm": bool(rec.alarm_active) if rec is not None else False,
                "fsm_state": rec.state.value if rec is not None else None,
            }
        self.frames.append(row)

    def run(self, seconds: float) -> "DriverSim":
        for _ in range(int(round(seconds / self.dt))):
            self.step()
        return self

    # ---- evaluation
    def scores(self, transition_tolerance_sec: float = 2.0, alarm_grace_sec: float = 3.0) -> Scores:
        s = Scores(runs=1)
        dt = self.dt
        # seat time as the reports see it: closed sessions (+ the ones closed now, at the end of the run)
        if not self._closed:
            self._closed = True
            self.events += [dict(e, emitted_ts=self.t) for e in self.fsm.close_all(self.t)]
        s.seat_fsm = sum(float(e.get("value") or 0.0) for e in self.events if e["type"] == "DRIVER_SESSION_END")
        for v in self.vehicles:
            rows = [(f["t"], f["vehicles"][v.vid]) for f in self.frames]
            # times of true onboard changes (boarding / dismount): the state machine may lag a little there
            changes = [rows[i][0] for i in range(1, len(rows))
                       if rows[i][1]["true_onboard"] != rows[i - 1][1]["true_onboard"]
                       or rows[i][1]["true_driver"] != rows[i - 1][1]["true_driver"]]

            def near_change(t: float) -> bool:
                return any(abs(t - c) <= transition_tolerance_sec for c in changes)

            far_true, alarm_fsm = [], []
            for t, r in rows:
                if not r["seen"]:
                    far_true.append(far_true[-1] if far_true else False)     # one missed detection: no change
                    alarm_fsm.append(r["fsm_alarm"])
                    continue
                if r["true_driver"] is not None:
                    s.t_driver += dt
                    if r["fsm_true"] == r["true_driver"]:
                        s.t_correct += dt
                        if not near_change(t):
                            s.onboard_total += dt
                            s.onboard_ok += dt if r["fsm_onboard"] == r["true_onboard"] else 0.0
                    elif r["fsm_driver"] is None:
                        if near_change(t):
                            s.t_driver -= dt                    # boarding / change in progress: not judged
                        else:
                            s.t_missing += dt
                    else:
                        s.t_wrong += dt
                else:
                    s.t_empty += dt
                    s.t_false += dt if r["fsm_driver"] is not None else 0.0
                s.seat_true += dt if r["true_onboard"] else 0.0
                away = (r["true_driver"] is not None and not r["true_onboard"]
                        and (not r["driver_in_view"] or (r["true_dist"] or 0.0) > self.safe_m))
                far_true.append(away)
                alarm_fsm.append(r["fsm_alarm"])
            times = [t for t, _ in rows]
            for a, b in _intervals(far_true):
                if times[b] - times[a] < alarm_grace_sec:
                    continue                                    # too short to demand an alarm
                s.alarms_true += 1
                hit = next((i for i in range(a, min(len(rows), b + 1)) if alarm_fsm[i]), None)
                if hit is not None:
                    s.alarms_found += 1
                    s.alarm_delay += max(0.0, times[hit] - times[a])
            for a, b in _intervals(alarm_fsm):
                lo = max(0, a - int(alarm_grace_sec / dt))
                near_far = False
                for i in range(lo, b + 1):
                    r = rows[i][1]
                    if far_true[i] or (r["true_dist"] is not None and r["true_dist"] > self.safe_m - 0.6):
                        near_far = True
                        break
                if not near_far:
                    s.alarms_false += 1
        return s


def _intervals(flags: List[bool]) -> List[Tuple[int, int]]:
    out, start = [], None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        elif not f and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(flags) - 1))
    return out


# ============================================================================================== scenarios
def _jit(sim: DriverSim, a: float = 0.4) -> float:
    return float(sim.rng.uniform(-a, a))


def _side_seen(sim: DriverSim, v: SimVehicle, prefer_visible: bool = True) -> int:
    """Which side of the vehicle faces the camera (+1 left / -1 right)."""
    left = v.to_world(np.array([[0.0, 1.0, 0.0]]))[0]
    right = v.to_world(np.array([[0.0, -1.0, 0.0]]))[0]
    near_left = np.linalg.norm(left - sim.cam.pos) < np.linalg.norm(right - sim.cam.pos)
    return (1 if near_left else -1) * (1 if prefer_visible else -1)


def sc_board_drive_alarm(sim: DriverSim, kind: str, heading: float) -> float:
    """Board, drive, stop, dismount, walk away (alarm), come back (cleared), board again, drive."""
    v = sim.vehicle(0.0 + _jit(sim), 4.0 + _jit(sim), heading, kind)
    d = sim.person(-5.0, 2.0 + _jit(sim), height=float(sim.rng.uniform(1.62, 1.85)))
    walker = sim.person(5.5, 8.5)
    side = _side_seen(sim, v, bool(sim.rng.random() < 0.6))

    def driver() -> Script:
        yield from d.board(sim, v, side)
        while v.speed == 0.0:
            yield
        while v.speed > 0.0 or v.x < 2.0:
            yield
        yield from d.wait(sim, 2.0)
        yield from d.dismount(sim, side)
        yield from d.walk_to(sim, d.x - 5.5, d.y - 1.0)
        yield from d.wait(sim, 6.0)
        yield from d.board(sim, v, side)

    def veh() -> Script:
        yield from v.wait_for_driver(sim, 2.0)
        yield from v.drive_to(sim, 3.5, 6.0)
        while v.driver is not None or sim.session_driver[v.vid] is None:
            yield                                              # wait for the dismount
        while v.driver is None:
            yield                                              # ... and for the re-boarding
        yield from v.wait(sim, 2.0)
        yield from v.drive_to(sim, -2.0, 3.0)

    def stroll() -> Script:
        yield from walker.walk_to(sim, -5.0, 8.0, 1.2)
        yield from walker.walk_to(sim, 5.0, 9.0, 1.2)

    sim.script(driver()), sim.script(veh()), sim.script(stroll())
    return 62.0


def sc_bystander_behind(sim: DriverSim, kind: str, heading: float) -> float:
    """Nobody drives at first: one person stands BEHIND the parked forklift, one walks past in front. Later the
    real driver boards and drives off while the bystander is still there."""
    v = sim.vehicle(0.0 + _jit(sim), 4.5 + _jit(sim), heading, kind)
    away = np.array([v.x, v.y]) - sim.cam.pos[:2]
    away /= np.linalg.norm(away)
    # right behind the vehicle as seen from the camera: 0.4 - 0.9 m beyond its far edge
    a_local = v.to_local(np.array([[v.x + away[0], v.y + away[1], 0.0]]))[0][:2]
    extent = float((np.array([(-1.2, -0.6), (2.2, -0.6), (2.2, 0.6), (-1.2, 0.6)]) * a_local).sum(axis=1).max())
    behind = np.array([v.x, v.y]) + away * (extent + float(sim.rng.uniform(0.4, 0.9)))
    front = np.array([v.x, v.y]) - away * 2.5
    b = sim.person(5.0, 7.5)
    passer = sim.person(-6.0, float(front[1]))
    d = sim.person(-5.5, 1.5)
    side = _side_seen(sim, v, True)

    def bystander() -> Script:
        yield from b.walk_to(sim, float(behind[0]), float(behind[1]))
        yield from b.wait(sim, 45.0)

    def passing() -> Script:
        yield from passer.wait(sim, 5.0)
        yield from passer.walk_to(sim, 6.0, float(front[1]), 1.2)
        yield from passer.walk_to(sim, -6.0, float(front[1]) - 0.5, 1.2)

    def driver() -> Script:
        yield from d.wait(sim, 24.0)
        yield from d.board(sim, v, side)

    def veh() -> Script:
        yield from v.wait_for_driver(sim, 2.5)
        yield from v.drive_to(sim, -3.5, 6.5)
        yield from v.drive_to(sim, 3.0, 7.0)

    sim.script(bystander()), sim.script(passing()), sim.script(driver()), sim.script(veh())
    return 52.0


def sc_drive_past_people(sim: DriverSim, kind: str, heading: float) -> float:
    """A driven forklift passes close to standing people and one who walks the same way."""
    v = sim.vehicle(-4.5, 4.0 + _jit(sim), heading, kind)
    d = sim.person(-6.0, 2.0)
    standing = [sim.person(-1.0 + 2.0 * i + _jit(sim, 0.2), 5.3 + _jit(sim, 0.2)) for i in range(3)]
    walker = sim.person(-3.0, 2.9)
    side = _side_seen(sim, v, True)

    def driver() -> Script:
        yield from d.board(sim, v, side)

    def veh() -> Script:
        yield from v.wait_for_driver(sim, 2.0)
        yield from v.drive_to(sim, 4.5, 4.1, 1.6)
        yield from v.wait(sim, 3.0)
        yield from v.drive_to(sim, -3.0, 4.2, 1.6)

    def same_way() -> Script:
        yield from walker.wait(sim, 6.0)
        yield from walker.walk_to(sim, 4.5, 3.0, 1.2)
        yield from walker.walk_to(sim, -4.0, 2.9, 1.2)

    sim.script(driver()), sim.script(veh()), sim.script(same_way())
    for p in standing:
        sim.script(p.wait(sim, 60.0))
    return 40.0


def sc_helper_at_forks(sim: DriverSim, kind: str, heading: float) -> float:
    """The driver stays seated while a helper works at the forks and next to the cab."""
    v = sim.vehicle(-3.0, 4.0 + _jit(sim), heading, kind)
    d = sim.person(-5.5, 2.0)
    h = sim.person(4.5, 7.0)
    side = _side_seen(sim, v, True)

    def driver() -> Script:
        yield from d.board(sim, v, side)

    def veh() -> Script:
        yield from v.wait_for_driver(sim, 2.0)
        yield from v.drive_to(sim, 1.5, 5.0)
        yield from v.wait(sim, 22.0)
        yield from v.drive_to(sim, -3.0, 6.5)

    def helper() -> Script:
        yield from h.wait(sim, 9.0)
        yield from h.follow(sim, v, (2.6, 0.9), 9.0)           # at the forks
        yield from h.follow(sim, v, (-0.2, -1.1 * side), 9.0)  # next to the cab, on the far side
        yield from h.walk_to(sim, 5.0, 8.0)

    sim.script(driver()), sim.script(veh()), sim.script(helper())
    return 46.0


def sc_driver_change(sim: DriverSim, kind: str, heading: float) -> float:
    """Driver A parks and leaves the camera (unattended forklift); B takes over."""
    v = sim.vehicle(-2.0, 4.0 + _jit(sim), heading, kind)
    a = sim.person(-5.5, 2.0)
    b = sim.person(5.5, 8.0)
    side = _side_seen(sim, v, True)

    def first() -> Script:
        yield from a.board(sim, v, side)
        while v.speed == 0.0:
            yield
        while v.speed > 0.0:
            yield
        yield from a.wait(sim, 1.5)
        yield from a.dismount(sim, side)
        yield from a.leave(sim, 14.0, 3.0)

    def second() -> Script:
        yield from b.wait(sim, 30.0)
        yield from b.board(sim, v, side)

    def veh() -> Script:
        yield from v.wait_for_driver(sim, 2.0)
        yield from v.drive_to(sim, 2.0, 5.0)
        while sim.session_driver[v.vid] != b.pid or not b.seated:
            yield
        yield from v.wait(sim, 2.0)
        yield from v.drive_to(sim, -3.0, 6.0)

    sim.script(first()), sim.script(second()), sim.script(veh())
    return 52.0


def sc_two_forklifts(sim: DriverSim, kind: str, heading: float) -> float:
    """Two forklifts side by side, two drivers; one stays near after dismounting, one walks away."""
    v1 = sim.vehicle(-2.2, 4.0, heading, kind)
    v2 = sim.vehicle(2.2, 4.3, heading, kind)
    a, b = sim.person(-6.0, 2.0), sim.person(6.0, 2.2)
    s1, s2 = _side_seen(sim, v1, True), _side_seen(sim, v2, True)

    def da() -> Script:
        yield from a.board(sim, v1, s1)
        while v1.speed == 0.0:
            yield
        while v1.speed > 0.0:
            yield
        yield from a.wait(sim, 1.5)
        yield from a.dismount(sim, s1)
        yield from a.follow(sim, v1, (2.8, 1.0), 20.0)         # works at the forks: no alarm

    def db() -> Script:
        yield from b.wait(sim, 2.0)
        yield from b.board(sim, v2, s2)
        while v2.speed == 0.0:
            yield
        while v2.speed > 0.0:
            yield
        yield from b.wait(sim, 1.5)
        yield from b.dismount(sim, s2)
        yield from b.walk_to(sim, b.x + 1.0, b.y - 5.5)
        yield from b.wait(sim, 12.0)

    def m1() -> Script:
        yield from v1.wait_for_driver(sim, 2.0)
        yield from v1.drive_to(sim, -3.5, 7.0)

    def m2() -> Script:
        yield from v2.wait_for_driver(sim, 2.0)
        yield from v2.drive_to(sim, 3.5, 7.2)

    sim.script(da()), sim.script(db()), sim.script(m1()), sim.script(m2())
    return 46.0


def sc_work_near_vehicle(sim: DriverSim, kind: str, heading: float) -> float:
    """Dismounted driver works around the forklift (no alarm), then goes away (alarm), then returns."""
    v = sim.vehicle(0.0, 4.5 + _jit(sim), heading, kind)
    d = sim.person(-5.0, 2.5)
    side = _side_seen(sim, v, True)

    def driver() -> Script:
        yield from d.board(sim, v, side)
        yield from d.wait(sim, 6.0)
        yield from d.dismount(sim, side)
        for local in ((2.9, 1.2), (2.9, -1.2), (0.0, 1.9 * side), (2.9, 1.2)):
            yield from d.follow(sim, v, local, 5.0)
        yield from d.walk_to(sim, v.x + 5.2, v.y + 2.5)
        yield from d.wait(sim, 7.0)
        yield from d.board(sim, v, side)

    sim.script(driver())
    sim.script(v.wait(sim, 200.0))
    return 58.0


def sc_group_one_boards(sim: DriverSim, kind: str, heading: float) -> float:
    """Three colleagues talk next to the parked forklift; one of them boards and drives off."""
    v = sim.vehicle(0.0, 4.5 + _jit(sim), heading, kind)
    side = _side_seen(sim, v, True)
    spot = v.to_world(np.array([[-0.3, 2.3 * side, 0.0], [0.8, 2.6 * side, 0.0], [-1.3, 2.7 * side, 0.0]]))
    ppl = [sim.person(float(spot[i][0]), float(spot[i][1])) for i in range(3)]
    d = ppl[int(sim.rng.integers(0, 3))]

    def driver() -> Script:
        yield from d.wait(sim, 8.0)
        yield from d.board(sim, v, side)

    def veh() -> Script:
        yield from v.wait_for_driver(sim, 2.0)
        yield from v.drive_to(sim, v.x + 4.0 * math.cos(v.heading) - 1.0, v.y + 2.5)
        yield from v.drive_to(sim, -3.0, 7.0)

    sim.script(driver()), sim.script(veh())
    for p in ppl:
        if p is not d:
            sim.script(p.wait(sim, 60.0))
    return 34.0


SCENARIOS: Dict[str, Callable[[DriverSim, str, float], float]] = {
    "board_drive_alarm": sc_board_drive_alarm,
    "bystander_behind": sc_bystander_behind,
    "drive_past_people": sc_drive_past_people,
    "helper_at_forks": sc_helper_at_forks,
    "driver_change": sc_driver_change,
    "two_forklifts": sc_two_forklifts,
    "work_near_vehicle": sc_work_near_vehicle,
    "group_one_boards": sc_group_one_boards,
}
KINDS = ("open", "roof", "cabin")
HEADINGS = (0.0, 90.0, 180.0, 270.0)


def run_case(scenario: str, kind: str, heading: float, camera: str, seed: int, noise: Optional[Noise] = None,
             config: Optional[Dict[str, Any]] = None, use_contour: bool = True) -> DriverSim:
    sim = DriverSim(CAMERAS[camera](), seed=seed, noise=noise, config=config, use_contour=use_contour)
    duration = SCENARIOS[scenario](sim, kind, heading)
    return sim.run(duration)


def run_benchmark(scenarios=None, kinds=KINDS, headings=HEADINGS, cameras=None, seeds=(0,), noise=None,
                  config=None, use_contour=True, progress: Optional[Callable[[str], None]] = None
                  ) -> Dict[Tuple[str, str], Scores]:
    """-> scores per (scenario, vehicle kind)."""
    out: Dict[Tuple[str, str], Scores] = {}
    for sc in scenarios or list(SCENARIOS):
        for kind in kinds:
            total = out.setdefault((sc, kind), Scores())
            for cam in cameras or list(CAMERAS):
                for h in headings:
                    for seed in seeds:
                        total.add(run_case(sc, kind, h, cam, seed, noise, config, use_contour).scores())
            if progress:
                progress(f"{sc:20s} {kind:6s} {total.summary()}")
    return out

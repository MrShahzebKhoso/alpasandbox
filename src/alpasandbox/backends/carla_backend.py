"""CARLA adapter (0.9.14+ Python API). Runs in synchronous mode at a fixed dt so the sim waits
for the model, however slow it is.

Frames: CARLA is left-handed (x forward, y right, yaw clockwise from above). alpasandbox uses
x forward, y left, yaw CCW, origin at the REAR AXLE. So y -> -y, yaw -> -yaw, and the ego pose
is shifted from CARLA's actor origin (body centre) to the rear axle measured from the wheels.

Maps:
  town="scenario"  (default) builds a road from the scenario (straight or S-curves) with
                   OpenDRIVE, so every scenario looks the same here as in the builtin backend.
  town="Town10HD_Opt" (or any map) uses that town; actors are placed along the ego's lane.
  town="current"   keeps whatever map the server has loaded.
"""
from __future__ import annotations

import math
import queue
import random

import numpy as np

from ..core import Control, EgoState, StepResult
from ..rig import REFERENCE_RIG
from ..scenario import Scenario
from .base import Backend
from .opendrive import generate_opendrive

BLUEPRINTS = {"pedestrian": "walker.pedestrian.0001", "vehicle": "vehicle.audi.a2",
              "cone": "static.prop.constructioncone"}
SPAWN_Z = {"pedestrian": 1.0, "vehicle": 0.5, "cone": 0.2}
POLYGON_KINDS = ("surface", "crosswalk")


def _boundary_kind(wp, neighbour, driving) -> str:
    """How to draw one side of a driving lane: the road edge, a lane divider, or the centre line."""
    if neighbour is None or neighbour.lane_type != driving:
        return "edge"
    return "center" if (neighbour.lane_id > 0) != (wp.lane_id > 0) else "lane"


def _crosswalk_polygons(points: list[tuple[float, float]]) -> list[list[tuple[float, float]]]:
    """CARLA returns crosswalks as one flat list of closed polygons (the first point repeats at the end)."""
    polys, cur = [], []
    for p in points:
        cur.append(p)
        if len(cur) > 2 and math.dist(p, cur[0]) < 1e-3:
            polys.append(cur)
            cur = []
    return polys


def lane_geometry(segments: list[dict], crosswalks: list) -> list[tuple[str, list]]:
    """Bird's-eye map from lane segments (ego frame: x forward, y left; pure, no CARLA needed).

    Each segment: ctr/left/right polylines, junction flag, and left_kind/right_kind from
    `_boundary_kind`. Every lane, junctions included, gets a filled "surface" so intersections read
    as road. Outside junctions: "edge" for the kerb, "center" between directions, and "lane" for
    dividers (drawn once, on each lane's right side). Inside junctions, only a faint turn path.
    """
    out = [("surface", list(sg["left"]) + list(reversed(sg["right"]))) for sg in segments if len(sg["ctr"]) > 1]
    out += [("crosswalk", list(p)) for p in crosswalks]
    for sg in segments:
        if len(sg["ctr"]) < 2:
            continue
        if sg["junction"]:
            out.append(("junction", list(sg["ctr"])))
            continue
        if sg["left_kind"] in ("edge", "center"):
            out.append((sg["left_kind"], list(sg["left"])))
        out.append((sg["right_kind"], list(sg["right"])))
    return out


class CarlaBackend(Backend):
    name = "carla"

    def __init__(self, host: str = "localhost", port: int = 2000, town: str = "scenario",
                 image_size=(640, 360), rig=REFERENCE_RIG, dt: float = 0.1,
                 ego_blueprint: str = "vehicle.tesla.model3", spawn_index: int = 0, traffic: int = 0,
                 tm_port: int = 8010, follow_spectator: bool = True, timeout: float = 60.0, seed: int = 0,
                 weather: str = "ClearNoon", start_speed: float | None = None):
        import carla  # imported lazily so the base install works without CARLA

        self.carla, self.rig, self.dt, self.image_size = carla, rig, dt, tuple(image_size)
        self.town, self.ego_bp, self.spawn_index = town, ego_blueprint, spawn_index
        self.traffic, self.tm_port, self.follow, self.rng = traffic, tm_port, follow_spectator, random.Random(seed)
        self.client = carla.Client(host, port)
        self.client.set_timeout(timeout)
        if town not in ("scenario", "current"):
            self.world = self.client.load_world(town)
        else:
            self.world = self.client.get_world()
        self._road_key = None
        self.weather, self.start_speed = weather, start_speed
        self._sync(True)
        self._apply_weather()
        self.spawned: list = []
        self.queues: dict[int, queue.Queue] = {}
        self.t, self.rear_offset, self._lines_cache = 0.0, -1.4, None

    # --- world management ---------------------------------------------------------------
    def _sync(self, on: bool) -> None:
        s = self.world.get_settings()
        s.synchronous_mode, s.fixed_delta_seconds = on, (self.dt if on else None)
        self.world.apply_settings(s)
        if self.traffic:
            try:
                self.client.get_trafficmanager(self.tm_port).set_synchronous_mode(on)
            except Exception:
                pass

    def _apply_weather(self) -> None:
        """Default ClearNoon: CARLA's default low sun washes out cameras and is far from the
        daytime footage Alpamayo was trained on. Any carla.WeatherParameters preset name works."""
        if not self.weather or self.weather == "default":
            return
        preset = getattr(self.carla.WeatherParameters, self.weather, None)
        if preset is None:
            raise ValueError(f"unknown CARLA weather preset {self.weather!r} (e.g. ClearNoon, CloudyNoon, WetNoon, ClearSunset)")
        self.world.set_weather(preset)

    def _ensure_world(self, sc: Scenario) -> None:
        if self.town != "scenario":
            return
        key = (round(sc.road_amplitude, 3), round(sc.road_wavelength, 3), round(sc.duration * sc.ego_speed))
        if key == self._road_key:
            return
        c = self.carla
        params = c.OpendriveGenerationParameters(vertex_distance=2.0, max_road_length=50.0, wall_height=0.0,
                                                 additional_width=1.0, smooth_junctions=True,
                                                 enable_mesh_visibility=True)
        self.world = self.client.generate_opendrive_world(generate_opendrive(sc), params)
        self._sync(True)
        self._apply_weather()
        self._road_key, self._lines_cache = key, None

    def _release_lights(self) -> None:
        if getattr(self, "_held", False):
            try:
                self.world.freeze_all_traffic_lights(False)
                self.world.reset_all_traffic_lights()
            except Exception:
                pass
            self._held = False

    def _cleanup(self) -> None:
        self._release_lights()
        # Reverse spawn order: scripted actors and traffic, then sensors, then the ego.
        for a in reversed(self.spawned):
            try:
                if hasattr(a, "stop") and a.type_id.startswith("sensor."):
                    a.stop()
                a.destroy()
            except Exception:
                pass
        self.spawned, self.queues = [], {}

    # --- frame conversions --------------------------------------------------------------
    def _to_ours(self, loc) -> tuple[float, float]:
        return loc.x, -loc.y

    def _rear_axle_offset(self) -> float:
        """Signed x of the rear-axle centre in the actor frame (metres), from wheel positions."""
        try:
            pc = self.ego.get_physics_control()
            tf = self.ego.get_transform()
            fwd = tf.get_forward_vector()
            xs = []
            for w in pc.wheels:                      # wheel.position is world-space, centimetres
                dx, dy = w.position.x / 100.0 - tf.location.x, w.position.y / 100.0 - tf.location.y
                xs.append(dx * fwd.x + dy * fwd.y)
            xs.sort()
            rear, front = (xs[0] + xs[1]) / 2, (xs[-1] + xs[-2]) / 2
            self.wheelbase = max(1.5, front - rear)
            self.max_steer_rad = math.radians(max(w.max_steer_angle for w in pc.wheels)) or 0.5
            return rear
        except Exception:
            return -1.4

    # --- lifecycle ----------------------------------------------------------------------
    def reset(self, scenario: Scenario) -> StepResult:
        c = self.carla
        self._cleanup()
        self._ensure_world(scenario)
        w, m = self.world, self.world.get_map()
        self.sc, self.t, self.collided, self.collided_with = scenario, 0.0, False, ""
        lib = w.get_blueprint_library()

        if self.town == "scenario":
            spawn_wp = m.get_waypoint_xodr(1, -1, 5.0) or m.get_waypoint(c.Location(5, 1.75, 0))
        else:
            points = m.get_spawn_points()
            spawn_wp = m.get_waypoint(points[self.spawn_index % len(points)].location)
        spawn = spawn_wp.transform
        spawn.location.z += 0.5
        bp = lib.find(self.ego_bp)
        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", "hero")
        self.ego = w.spawn_actor(bp, spawn)
        self.spawned.append(self.ego)
        w.tick()
        self.rear_offset = self._rear_axle_offset()

        for cam in self.cameras():
            cbp = lib.find("sensor.camera.rgb")
            cbp.set_attribute("image_size_x", str(self.image_size[0]))
            cbp.set_attribute("image_size_y", str(self.image_size[1]))
            cbp.set_attribute("fov", str(cam.fov_deg))
            tf = c.Transform(c.Location(x=cam.x + self.rear_offset, y=-cam.y, z=cam.z),
                             c.Rotation(yaw=-cam.yaw_deg, pitch=cam.pitch_deg))
            sensor = w.spawn_actor(cbp, tf, attach_to=self.ego)
            q: queue.Queue = queue.Queue()
            sensor.listen(q.put)
            self.spawned.append(sensor)
            self.queues[cam.cam_id] = q
        col = w.spawn_actor(lib.find("sensor.other.collision"), c.Transform(), attach_to=self.ego)
        col.listen(self._on_collision)
        self.spawned.append(col)

        # Scripted actors: a.x metres along the ego lane, a.y metres to its left.
        self.actors, self.unplaced = [], []
        for a in scenario.actors:
            wps = spawn_wp.next(a.x) if a.x > 0 else spawn_wp.previous(max(-a.x, 0.1))
            if not wps:
                continue
            ref = wps[0].transform
            right = ref.get_right_vector()
            loc = c.Location(ref.location.x - right.x * a.y, ref.location.y - right.y * a.y,
                             ref.location.z + SPAWN_Z[a.kind])
            actor = None
            for dz in (0.0, 0.5, 1.0):         # sidewalks and kerbs: retry a little higher before giving up
                loc.z = ref.location.z + SPAWN_Z[a.kind] + dz
                actor = w.try_spawn_actor(lib.find(BLUEPRINTS[a.kind]), c.Transform(loc, ref.rotation))
                if actor:
                    break
            if not actor:
                self.unplaced.append(f"{a.kind} at {a.x:.0f} m ahead, {a.y:+.1f} m")
            if actor:
                self.spawned.append(actor)
                self.actors.append({"spec": a, "actor": actor, "fwd": ref.get_forward_vector(), "right": right,
                                    "speed": math.hypot(a.vx, a.vy)})
        if self.traffic:
            self._spawn_traffic(lib, m)

        # Town spawn points often sit at junctions, so start from rest there: a car launched at a stop
        # line is not a situation any driver faces, and a standing start also makes the padded
        # history at t=0 exactly right (stationary car, identical frames). Scenario roads keep ego_speed.
        speed = self.start_speed if self.start_speed is not None else (
            scenario.ego_speed if self.town == "scenario" else 0.0)
        fwd = spawn.get_forward_vector()
        self.ego.set_target_velocity(c.Vector3D(fwd.x * speed, fwd.y * speed, 0))
        return self._observe(tick=True)

    def _spawn_traffic(self, lib, m) -> None:
        try:
            tm = self.client.get_trafficmanager(self.tm_port)
        except RuntimeError as exc:
            raise RuntimeError(f"CARLA Traffic Manager could not open port {self.tm_port} on this machine ({exc}). "
                               "Pick a free one with [carla] tm_port, or run with --traffic 0.") from exc
        tm.set_synchronous_mode(True)
        ego_loc = self.ego.get_location()
        points = [p for p in m.get_spawn_points() if p.location.distance(ego_loc) > 15]
        self.rng.shuffle(points)
        cars = [b for b in lib.filter("vehicle.*") if int(b.get_attribute("number_of_wheels").as_int()) == 4]
        self.traffic_spawned = 0
        for p in points[: self.traffic]:      # at most one car per free spawn point
            v = self.world.try_spawn_actor(self.rng.choice(cars), p)
            if v:
                v.set_autopilot(True, tm.get_port())
                self.spawned.append(v)
                self.traffic_spawned += 1

    def _on_collision(self, event) -> None:
        self.collided = True
        self.collided_with = getattr(event.other_actor, "type_id", "")

    def step(self, ctrl: Control) -> StepResult:
        c = self.carla
        self.ego.apply_control(c.VehicleControl(steer=float(-ctrl.steer), throttle=float(ctrl.throttle),
                                                brake=float(ctrl.brake)))
        for st in self.actors:   # scripted motion, no AI
            a, actor = st["spec"], st["actor"]
            if self.t < a.start_t or not (a.vx or a.vy):
                continue
            if a.decel and self.t >= a.decel_start_t:
                st["speed"] = max(0.0, st["speed"] - a.decel * self.dt)
            base = math.hypot(a.vx, a.vy) or 1.0
            k = st["speed"] / base
            fwd, right = st["fwd"], st["right"]
            vx, vy = a.vx * k, a.vy * k
            lat_start = a.start_t if a.lateral_start_t is None else a.lateral_start_t
            if self.t < lat_start or st.get("lat_done"):
                vy = 0.0
            elif a.y_target is not None:
                st["lat"] = st.get("lat", a.y) + vy * self.dt
                if (vy > 0 and st["lat"] >= a.y_target) or (vy < 0 and st["lat"] <= a.y_target):
                    st["lat_done"], vy = True, 0.0
            v = c.Vector3D(fwd.x * vx - right.x * vy, fwd.y * vx - right.y * vy, 0.0)
            if a.kind == "pedestrian":
                spd = math.hypot(v.x, v.y)
                direction = c.Vector3D(v.x / spd, v.y / spd, 0.0) if spd > 1e-3 else c.Vector3D(0, 0, 0)
                actor.apply_control(c.WalkerControl(direction=direction, speed=spd))
            else:
                actor.set_target_velocity(v)
        self.t += self.dt
        return self._observe(tick=True)

    def _observe(self, tick: bool) -> StepResult:
        frame = self.world.tick() if tick else None
        images = {}
        for cid, q in self.queues.items():
            img = q.get(timeout=30.0)
            while frame is not None and img.frame < frame:   # drop stale frames
                img = q.get(timeout=30.0)
            arr = np.frombuffer(img.raw_data, np.uint8).reshape(img.height, img.width, 4)
            images[cid] = np.ascontiguousarray(arr[:, :, 2::-1])  # BGRA -> RGB
        tf, vel = self.ego.get_transform(), self.ego.get_velocity()
        fwd = tf.get_forward_vector()
        rx, ry = tf.location.x + fwd.x * self.rear_offset, tf.location.y + fwd.y * self.rear_offset
        yaw = -math.radians(tf.rotation.yaw)
        ego = EgoState(self.t, rx, -ry, yaw, math.hypot(vel.x, vel.y))
        m = self.world.get_map()
        lane_wp = m.get_waypoint(tf.location, project_to_road=False, lane_type=self.carla.LaneType.Driving)
        near_wp = m.get_waypoint(tf.location)
        r = near_wp.transform.get_right_vector()
        dx, dy = tf.location.x - near_wp.transform.location.x, tf.location.y - near_wp.transform.location.y
        lane_offset = -(dx * r.x + dy * r.y)          # left positive
        if self.follow:
            self._move_spectator(tf)
        collided, self.collided = self.collided, False
        priv = {"ego": ego, "actors": self._nearby_actors(tf.location), "lines": self._lines(rx, -ry),
                "collided_with": self.collided_with if collided else "", "traffic_light": self._signal()}
        return StepResult(self.t, ego, images, collision=collided, off_road=lane_wp is None,
                          lane_offset=lane_offset, done=self.t >= self.sc.duration - 1e-9, privileged=priv)

    def _ego_light(self):
        """The light governing the ego: the one it is stopped at, else the next one on its lane (60 m)."""
        try:
            if self.ego.is_at_traffic_light():
                return self.ego.get_traffic_light()
            wp = self.world.get_map().get_waypoint(self.ego.get_location())
            for lm in wp.get_landmarks_of_type(60.0, "1000001", False):   # OpenDRIVE code for signals
                tl = self.world.get_traffic_light(lm)
                if tl is not None:
                    return tl
        except Exception:
            pass
        return None

    def debug_state(self) -> dict:
        """What CARLA actually applied after the last tick, plus the light state the ego itself sees."""
        try:
            c, tl = self.ego.get_control(), self._ego_light()
            return {"applied": {"throttle": round(c.throttle, 3), "brake": round(c.brake, 3),
                                "steer": round(c.steer, 4), "hand_brake": c.hand_brake, "reverse": c.reverse,
                                "manual_gear_shift": c.manual_gear_shift, "gear": c.gear},
                    "ego_tl_state": str(self.ego.get_traffic_light_state()).split(".")[-1],
                    "at_light": self.ego.is_at_traffic_light(),
                    "light_id": tl.id if tl is not None else None,
                    "ego_light_id": (self.ego.get_traffic_light().id
                                     if self.ego.get_traffic_light() is not None else None),
                    "stop_line_m": self._stop_line_distance(tl)}
        except Exception as exc:
            return {"error": repr(exc)}

    def _stop_line_distance(self, tl) -> float | None:
        """Metres from the ego's front bumper to its light's stop line in the ego's lane (< 0: past it)."""
        if tl is None:
            return None
        tf = self.ego.get_transform()
        f = tf.get_forward_vector()
        bb = self.ego.bounding_box
        best = None
        for w in tl.get_stop_waypoints():
            dx, dy = w.transform.location.x - tf.location.x, w.transform.location.y - tf.location.y
            lon, lat = dx * f.x + dy * f.y, abs(-dx * f.y + dy * f.x)
            if lat < 2.0 and (best is None or lat < best[1]):
                best = (lon - bb.location.x - bb.extent.x, lat)
        return None if best is None else round(best[0], 2)

    def set_signal(self, state: str | None) -> str:
        """Force the ego's light to Red/Yellow/Green and hold it, or None to hand it back to CARLA."""
        tl = self._ego_light()
        if tl is None:
            raise RuntimeError("no traffic light ahead of the ego within 60 m")
        if state is None:
            tl.freeze(False)
            return "released"
        tl.set_state(getattr(self.carla.TrafficLightState, state.capitalize()))
        tl.freeze(True)
        self._held = True
        return state.capitalize()

    def _signal(self) -> str | None:
        """Ground-truth state of the light governing the ego (Red/Yellow/Green), or None."""
        tl = self._ego_light()
        if tl is None:
            return None
        try:
            return str(tl.get_state()).split(".")[-1]
        except Exception:
            return None

    def _move_spectator(self, tf) -> None:
        c = self.carla
        fwd = tf.get_forward_vector()
        loc = c.Location(tf.location.x - fwd.x * 9, tf.location.y - fwd.y * 9, tf.location.z + 4.5)
        try:
            self.world.get_spectator().set_transform(c.Transform(loc, c.Rotation(pitch=-18, yaw=tf.rotation.yaw)))
        except Exception:
            pass

    def _nearby_actors(self, at, radius: float = 80.0) -> list[dict]:
        out = []
        for a in self.world.get_actors():
            tid = a.type_id
            kind = ("vehicle" if tid.startswith("vehicle.") else "pedestrian" if tid.startswith("walker.")
                    else "cone" if "cone" in tid else None)
            if kind is None or a.id == self.ego.id:
                continue
            loc = a.get_location()
            if loc.distance(at) > radius:
                continue
            ext, v, tf = a.bounding_box.extent, a.get_velocity(), a.get_transform()
            out.append({"id": a.id, "kind": kind, "x": loc.x, "y": -loc.y, "yaw": -math.radians(tf.rotation.yaw),
                        "size": (2 * ext.x, 2 * ext.y, 2 * ext.z), "vx": v.x, "vy": -v.y})
        return out

    def _lines(self, x: float, y: float, radius: float = 110.0) -> list[dict]:
        if self._lines_cache is None:
            m, driving = self.world.get_map(), self.carla.LaneType.Driving
            segments = []
            for start, _end in m.get_topology():
                wps = [start] + start.next_until_lane_end(2.0)
                ctr, left, right = [], [], []
                for wp in wps:
                    loc, rv, half = wp.transform.location, wp.transform.get_right_vector(), wp.lane_width / 2
                    ctr.append((loc.x, -loc.y))
                    left.append((loc.x - rv.x * half, -(loc.y - rv.y * half)))
                    right.append((loc.x + rv.x * half, -(loc.y + rv.y * half)))
                segments.append({"ctr": ctr, "left": left, "right": right, "junction": start.is_junction,
                                 "left_kind": _boundary_kind(start, start.get_left_lane(), driving),
                                 "right_kind": _boundary_kind(start, start.get_right_lane(), driving)})
            try:
                crosswalks = _crosswalk_polygons([(p.x, -p.y) for p in m.get_crosswalks()])
            except Exception:
                crosswalks = []
            self._lines_cache = [(k, np.asarray(pts, float)) for k, pts in lane_geometry(segments, crosswalks)]
        out = []
        for kind, pts in self._lines_cache:
            near = np.hypot(pts[:, 0] - x, pts[:, 1] - y) < radius
            if near.any():   # polygons go whole (cutting points out would distort them); polylines are trimmed
                keep = pts if kind in POLYGON_KINDS else pts[near]
                out.append({"kind": kind, "pts": keep.round(2).tolist()})
        return out

    def close(self) -> None:
        self._cleanup()
        try:
            self._sync(False)
        except Exception:
            pass

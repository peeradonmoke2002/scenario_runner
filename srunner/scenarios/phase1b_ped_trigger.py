#!/usr/bin/env python3
"""
Phase1BPedTrigger — CARLA-only pedestrian trigger repeatability scenario.

This is a Phase 1B-only ScenarioRunner scenario. It reuses the same blind-spot
parked-bus geometry and lane-projected trigger logic as BtParkedWithBlindSpotPed,
but removes ROS 2, Autoware, and BT from the loop:

  - ScenarioRunner spawns/resets the ego, parked bus, and walker.
  - A custom py_trees behavior applies identical open-loop ego throttle every tick.
  - The walker starts when the ego reaches the real lane-projected trigger condition.
  - Per-tick event-grade rows are logged to CSV for repeatability analysis.

Parameters are read from XML <other_parameters>.
"""

import csv
import math
import os
import time

import carla
import py_trees

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime
from srunner.scenarios.basic_scenario import BasicScenario


PED_ONSET_SPEED = 0.1
DEFAULT_OUT_CSV = (
    "/home/peeradon/autoware/src/av-stack-playground/bt_obstacle_eval/"
    "analysis/determinism/phase1B_ped_trigger/run_01.csv"
)


def get_value_parameter(config, name, p_type, default):
    if name in config.other_parameters:
        return p_type(config.other_parameters[name]['value'])
    return default


def get_location_on_same_road(start_wp, distance, step=1.0):
    """Follow waypoints up to distance but stop if road_id changes."""
    traveled = 0.0
    wp = start_wp
    while traveled < distance:
        nexts = wp.next(step)
        if not nexts:
            break
        wp_new = nexts[0]
        if wp_new.road_id != start_wp.road_id:
            break
        traveled += wp_new.transform.location.distance(wp.transform.location)
        wp = wp_new
    return wp.transform.location, traveled


def speed_of(actor):
    v = actor.get_velocity()
    return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


def look_at_rotation(camera_location, target_location):
    """Return a CARLA rotation that points from camera_location to target_location."""
    dx = target_location.x - camera_location.x
    dy = target_location.y - camera_location.y
    dz = target_location.z - camera_location.z
    horizontal = math.sqrt(dx * dx + dy * dy)
    yaw = math.degrees(math.atan2(dy, dx))
    pitch = math.degrees(math.atan2(dz, horizontal))
    return carla.Rotation(pitch=pitch, yaw=yaw)


def chase_spectator(actor):
    """Move the CARLA spectator camera behind and above the ego vehicle."""
    world = CarlaDataProvider.get_world()
    tf = actor.get_transform()
    fwd = tf.get_forward_vector()
    cam_loc = carla.Location(
        x=tf.location.x - 8.0 * fwd.x,
        y=tf.location.y - 8.0 * fwd.y,
        z=tf.location.z + 4.0,
    )
    cam = carla.Transform(cam_loc, look_at_rotation(cam_loc, tf.location))
    world.get_spectator().set_transform(cam)
    return cam


def overview_spectator(target_location):
    """Move the CARLA spectator camera to a fixed overview of the crossing zone."""
    world = CarlaDataProvider.get_world()
    target = carla.Location(
        x=target_location.x,
        y=target_location.y,
        z=target_location.z + 1.0,
    )
    cam_loc = carla.Location(
        x=target_location.x - 30.0,
        y=target_location.y - 22.0,
        z=34.0,
    )
    cam = carla.Transform(cam_loc, look_at_rotation(cam_loc, target))
    world.get_spectator().set_transform(cam)
    return cam


def top_spectator(target_location, z=45.0):
    """Move the CARLA spectator camera directly above the crossing zone."""
    world = CarlaDataProvider.get_world()
    cam = carla.Transform(
        carla.Location(
            x=target_location.x,
            y=target_location.y,
            z=z,
        ),
        carla.Rotation(pitch=-90.0, yaw=0.0),
    )
    world.get_spectator().set_transform(cam)
    return cam


class OpenLoopPedTrigger(py_trees.behaviour.Behaviour):
    """Drive ego open-loop, trigger walker, and log Phase 1B event fields."""

    def __init__(self, ego, walker, collision_location, trigger_from_center, wmap,
                 ticks, throttle, steer, ped_speed, settle_ticks, out_csv,
                 camera_mode, realtime, pace_seconds, stop_ego_after_trigger,
                 freeze_ego_after_trigger, name="OpenLoopPedTrigger"):
        super().__init__(name)
        self._ego = ego
        self._walker = walker
        self._collision_location = collision_location
        self._trigger_from_center = trigger_from_center
        self._wmap = wmap
        self._ticks = ticks
        self._throttle = throttle
        self._steer = steer
        self._ped_speed = ped_speed
        self._settle_ticks = settle_ticks
        self._out_csv = out_csv
        self._camera_mode = camera_mode
        self._realtime = realtime
        self._pace_seconds = pace_seconds
        self._stop_ego_after_trigger = stop_ego_after_trigger
        self._freeze_ego_after_trigger = freeze_ego_after_trigger
        self._ego_frozen = False
        self._last_wall = None

        self._i = 0
        self._triggered = False
        self._t_trigger = float("nan")
        self._t_ped_start = float("nan")
        self._csv_file = None
        self._csv = None
        self._walker_dir = None

    def _update_camera(self):
        if self._camera_mode == "chase":
            chase_spectator(self._ego)
        elif self._camera_mode == "overview":
            overview_spectator(self._collision_location)
        elif self._camera_mode == "top":
            top_spectator(self._collision_location)

    def _resolve_out_csv(self):
        root, ext = os.path.splitext(self._out_csv)
        if ext.lower() != ".csv":
            return self._out_csv
        if not os.path.exists(self._out_csv):
            return self._out_csv

        base_root = root
        start_i = 2
        stem = os.path.basename(root)
        if len(stem) >= 3 and stem[-3] == "_" and stem[-2:].isdigit():
            base_root = root[:-3]
            start_i = int(stem[-2:]) + 1

        i = start_i
        while True:
            candidate = f"{base_root}_{i:02d}{ext}"
            if not os.path.exists(candidate):
                return candidate
            i += 1

    def initialise(self):
        out_csv = self._resolve_out_csv()
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        self._csv_file = open(out_csv, "w", newline="")
        self._csv = csv.writer(self._csv_file)
        self._csv.writerow([
            "tick", "sim_t",
            "ped_x", "ped_y", "ped_z", "ped_yaw", "ped_vx", "ped_vy", "ped_speed", "is_moving",
            "trigger_flag", "t_trigger", "t_ped_start",
            "ego_x", "ego_y", "ego_yaw", "ego_speed", "road_dist",
        ])
        print(f"[Phase1BPedTrigger] logging to {out_csv}")
        self._walker_dir = self._walker.get_transform().get_forward_vector()
        self._ego.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        self._walker.apply_control(carla.WalkerControl(direction=self._walker_dir, speed=0.0))
        self._update_camera()
        print(f"[Phase1BPedTrigger] camera_mode={self._camera_mode}")
        self._last_wall = time.perf_counter()

    def _pace(self):
        if not self._realtime:
            return
        now = time.perf_counter()
        if self._last_wall is not None:
            elapsed = now - self._last_wall
            if elapsed < self._pace_seconds:
                time.sleep(self._pace_seconds - elapsed)
        self._last_wall = time.perf_counter()

    def update(self):
        self._pace()
        if self._i < self._settle_ticks:
            self._ego.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
            self._walker.apply_control(carla.WalkerControl(direction=self._walker_dir, speed=0.0))
            self._update_camera()
            self._i += 1
            return py_trees.common.Status.RUNNING

        tick = self._i - self._settle_ticks
        if tick >= self._ticks:
            return py_trees.common.Status.SUCCESS

        ego_tf = self._ego.get_transform()
        ego_wp = self._wmap.get_waypoint(ego_tf.location)
        road_dist = ego_wp.transform.location.distance(self._collision_location)
        if not self._triggered and road_dist <= self._trigger_from_center:
            self._triggered = True
            self._t_trigger = GameTime.get_carla_time()
            print(f"[Phase1BPedTrigger] trigger fired tick={tick} "
                  f"road_dist={road_dist:.3f} ego_speed={speed_of(self._ego):.3f}")

        if self._triggered and self._stop_ego_after_trigger:
            if self._freeze_ego_after_trigger and not self._ego_frozen:
                self._ego.set_target_velocity(carla.Vector3D(0, 0, 0))
                self._ego.set_target_angular_velocity(carla.Vector3D(0, 0, 0))
                self._ego_frozen = True
                print("[Phase1BPedTrigger] ego frozen after trigger to prevent bus collision")
            self._ego.apply_control(carla.VehicleControl(
                throttle=0.0, steer=0.0, brake=1.0, hand_brake=True))
        else:
            self._ego.apply_control(carla.VehicleControl(
                throttle=self._throttle, steer=self._steer, brake=0.0))

        self._walker.apply_control(carla.WalkerControl(
            direction=self._walker_dir,
            speed=self._ped_speed if self._triggered else 0.0))

        walker_tf = self._walker.get_transform()
        walker_v = self._walker.get_velocity()
        walker_speed = math.sqrt(walker_v.x ** 2 + walker_v.y ** 2 + walker_v.z ** 2)
        is_moving = walker_speed > PED_ONSET_SPEED
        if is_moving and math.isnan(self._t_ped_start):
            self._t_ped_start = GameTime.get_carla_time()

        self._csv.writerow([
            tick, f"{GameTime.get_carla_time():.6f}",
            f"{walker_tf.location.x:.6f}", f"{walker_tf.location.y:.6f}", f"{walker_tf.location.z:.6f}",
            f"{walker_tf.rotation.yaw:.6f}",
            f"{walker_v.x:.6f}", f"{walker_v.y:.6f}", f"{walker_speed:.6f}", int(is_moving),
            int(self._triggered), f"{self._t_trigger:.6f}", f"{self._t_ped_start:.6f}",
            f"{ego_tf.location.x:.6f}", f"{ego_tf.location.y:.6f}", f"{ego_tf.rotation.yaw:.6f}",
            f"{speed_of(self._ego):.6f}", f"{road_dist:.6f}",
        ])
        self._csv_file.flush()

        self._update_camera()

        self._i += 1
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None


class Phase1BPedTrigger(BasicScenario):
    """CARLA-only Phase 1B scenario using the real blind-spot pedestrian geometry."""

    def __init__(self, world, ego_vehicles, config, randomize=False,
                 debug_mode=False, criteria_enable=True, timeout=120):

        self._wmap = CarlaDataProvider.get_map()
        self._reference_waypoint = self._wmap.get_waypoint(config.trigger_points[0].location)
        self.timeout = get_value_parameter(config, "timeout", float, float(timeout))

        self._parked_dist = get_value_parameter(config, "parked_dist", float, 100.0)
        self._park_side = get_value_parameter(config, "park_side", str, "right")
        if self._park_side not in ("left", "right"):
            raise ValueError(f"'park_side' must be either 'right' or 'left' but '{self._park_side}' was given")
        self._yaw_offset = get_value_parameter(config, "yaw_offset", float, 0.0)
        self._adversary_speed = get_value_parameter(config, "adversary_speed", float, 2.0)
        self._trigger_dist = get_value_parameter(config, "trigger_dist", float, 4.0)

        self._ticks = get_value_parameter(config, "ticks", int, 350)
        self._throttle = get_value_parameter(config, "throttle", float, 0.5)
        self._steer = get_value_parameter(config, "steer", float, 0.0)
        self._settle_ticks = get_value_parameter(config, "settle_ticks", int, 20)
        self._walker_filter = get_value_parameter(config, "walker_filter", str, "walker.pedestrian.0001")
        self._out_csv = get_value_parameter(config, "out_csv", str, DEFAULT_OUT_CSV)
        self._camera_mode = get_value_parameter(config, "camera_mode", str, "off").lower()
        if get_value_parameter(config, "chase_view", str, "false").lower() == "true":
            self._camera_mode = "chase"
        if self._camera_mode not in ("top", "overview", "chase", "off"):
            raise ValueError("Phase1BPedTrigger camera_mode must be top, overview, chase, or off")
        self._realtime = get_value_parameter(config, "realtime", str, "false").lower() == "true"
        self._pace_seconds = get_value_parameter(config, "pace_seconds", float, 0.05)
        self._stop_ego_after_trigger = (
            get_value_parameter(config, "stop_ego_after_trigger", str, "true").lower() == "true")
        self._freeze_ego_after_trigger = (
            get_value_parameter(config, "freeze_ego_after_trigger", str, "true").lower() == "true")
        self._snap_ego_to_lane_center = (
            get_value_parameter(config, "snap_ego_to_lane_center", str, "true").lower() == "true")

        self._parked_transform = None
        self._ped_transform = None
        self._collision_wp = None
        self._parked_car_half_len = None
        self._trigger_from_center = None

        super().__init__(
            "Phase1BPedTrigger",
            ego_vehicles,
            config,
            world,
            debug_mode,
            criteria_enable=False,
        )

    def _get_blocker_transform(self, waypoint):
        if waypoint.lane_type == carla.LaneType.Sidewalk:
            new_location = waypoint.transform.location
        else:
            vector = waypoint.transform.get_right_vector()
            if self._park_side == "left":
                vector *= -1
            offset_location = carla.Location(
                waypoint.lane_width * 0.7 * vector.x,
                waypoint.lane_width * 0.7 * vector.y)
            new_location = waypoint.transform.location + offset_location
        new_location.z += 0.2

        new_rotation = carla.Rotation(
            pitch=waypoint.transform.rotation.pitch,
            yaw=waypoint.transform.rotation.yaw + self._yaw_offset,
            roll=waypoint.transform.rotation.roll,
        )
        return carla.Transform(new_location, new_rotation)

    def _get_walker_transform(self, waypoint):
        new_rotation = waypoint.transform.rotation
        new_rotation.yaw += 270 if self._park_side == "right" else 90

        if waypoint.lane_type == carla.LaneType.Sidewalk:
            new_location = waypoint.transform.location
        else:
            vector = waypoint.transform.get_right_vector()
            if self._park_side == "left":
                vector *= -1
            offset_location = carla.Location(waypoint.lane_width * vector.x, waypoint.lane_width * vector.y)
            new_location = waypoint.transform.location + offset_location
        new_location.z += 1.2

        return carla.Transform(new_location, new_rotation)

    def _initialize_actors(self, config):
        start_tf = config.trigger_points[0]
        if self._snap_ego_to_lane_center:
            wp_tf = self._reference_waypoint.transform
            start_tf = carla.Transform(
                carla.Location(wp_tf.location.x, wp_tf.location.y, config.trigger_points[0].location.z),
                carla.Rotation(
                    pitch=wp_tf.rotation.pitch,
                    yaw=wp_tf.rotation.yaw,
                    roll=wp_tf.rotation.roll,
                ),
            )
        self.ego_vehicles[0].set_transform(start_tf)
        self.ego_vehicles[0].set_target_velocity(carla.Vector3D(0, 0, 0))
        self.ego_vehicles[0].set_target_angular_velocity(carla.Vector3D(0, 0, 0))

        park_location, _ = get_location_on_same_road(self._reference_waypoint, self._parked_dist)
        park_wp = self._wmap.get_waypoint(park_location)
        self._parked_transform = self._get_blocker_transform(park_wp)
        self.parking_slots.append(self._parked_transform.location)

        parked_vehicle = CarlaDataProvider.request_new_actor(
            "vehicle.mitsubishi.fusorosa",
            self._parked_transform,
            rolename="scenario",
            attribute_filter={},
        )
        if parked_vehicle is None:
            raise ValueError("Phase1BPedTrigger: failed to spawn parked vehicle")

        parked_vehicle.apply_control(carla.VehicleControl(hand_brake=True))
        parked_vehicle.set_simulate_physics(True)
        self._parked_car_half_len = parked_vehicle.bounding_box.extent.x
        self.other_actors.append(parked_vehicle)

        walker_dist = parked_vehicle.bounding_box.extent.x + 0.5
        wps = park_wp.next(walker_dist)
        if not wps:
            raise ValueError("Phase1BPedTrigger: could not find walker waypoint")
        walker_wp = wps[0]
        self._collision_wp = walker_wp
        self._ped_transform = self._get_walker_transform(walker_wp)
        self.parking_slots.append(self._ped_transform.location)

        walker = CarlaDataProvider.request_new_actor(self._walker_filter, self._ped_transform)
        if walker is None:
            raise ValueError("Phase1BPedTrigger: failed to spawn pedestrian")
        walker.set_simulate_physics(True)
        self.other_actors.append(walker)

        ego_half_len = self.ego_vehicles[0].bounding_box.extent.x
        self._trigger_from_center = (
            self._trigger_dist + ego_half_len + 2 * self._parked_car_half_len + 0.5)

        print(
            "[Phase1BPedTrigger] "
            f"ego_start=({start_tf.location.x:.2f}, {start_tf.location.y:.2f}, yaw={start_tf.rotation.yaw:.1f}) "
            f"bus=({self._parked_transform.location.x:.2f}, {self._parked_transform.location.y:.2f}) "
            f"ped=({self._ped_transform.location.x:.2f}, {self._ped_transform.location.y:.2f}) "
            f"trigger_radius={self._trigger_from_center:.2f}m "
            f"out_csv={self._out_csv}"
        )
        if self._camera_mode == "top":
            cam = top_spectator(self._collision_wp.transform.location)
        elif self._camera_mode == "overview":
            cam = overview_spectator(self._collision_wp.transform.location)
        elif self._camera_mode == "chase":
            cam = chase_spectator(self.ego_vehicles[0])
        else:
            cam = None
        if cam is not None:
            print(
                "[Phase1BPedTrigger] spectator camera "
                f"mode={self._camera_mode} "
                f"loc=({cam.location.x:.2f}, {cam.location.y:.2f}, {cam.location.z:.2f}) "
                f"rot=(pitch={cam.rotation.pitch:.1f}, yaw={cam.rotation.yaw:.1f})"
            )
        world = CarlaDataProvider.get_world()
        world.debug.draw_string(self._parked_transform.location + carla.Location(z=3.0),
                                "PHASE1B BUS", draw_shadow=True,
                                color=carla.Color(255, 160, 0), life_time=30.0)
        world.debug.draw_string(self._ped_transform.location + carla.Location(z=2.0),
                                "PHASE1B PED START", draw_shadow=True,
                                color=carla.Color(0, 255, 255), life_time=30.0)

    def _setup_scenario_trigger(self, config):
        # BasicScenario's default trigger waits for ego time-to-arrival at the start point.
        # This Phase 1B scenario itself drives the ego open-loop, so that default trigger
        # would deadlock before OpenLoopPedTrigger can apply throttle.
        return None

    def _create_behavior(self):
        return OpenLoopPedTrigger(
            self.ego_vehicles[0],
            self.other_actors[1],
            self._collision_wp.transform.location,
            self._trigger_from_center,
            self._wmap,
            self._ticks,
            self._throttle,
            self._steer,
            self._adversary_speed,
            self._settle_ticks,
            self._out_csv,
            self._camera_mode,
            self._realtime,
            self._pace_seconds,
            self._stop_ego_after_trigger,
            self._freeze_ego_after_trigger,
        )

    def _create_test_criteria(self):
        return []

#!/usr/bin/env python3
"""
Phase1BPedTrigger — CARLA-only pedestrian trigger repeatability scenario.

A Phase 1B variant of BtParkedWithBlindSpotPed, structured the same way as its sibling
phase2_ros_controller.py: it SUBCLASSES BtParkedWithBlindSpotPed and reuses the parent's
geometry helpers (_get_blocker_transform / _get_walker_transform / get_location_on_same_road /
get_value_parameter / __del__), overriding only what Phase 1B needs.

Phase 1B removes ROS 2, Autoware, and BT from the loop — it is CARLA-only:

  - ScenarioRunner spawns/resets the ego, parked bus, and walker (_initialize_actors).
  - The custom OpenLoopPedTrigger behavior drives the ego open-loop (constant throttle),
    fires the lane-projected pedestrian trigger, walks the pedestrian, and logs per-tick
    event-grade rows to CSV for repeatability analysis — all in ONE behavior so each CSV row
    samples ego and walker at the same tick.

Because the ego is driven by the scenario (no external controller exists in Phase 1B), the
ego-drive + logging stay in OpenLoopPedTrigger rather than being decomposed into stock atoms.

Parameters are read from XML <other_parameters>. parked_dist=100 / trigger_dist=4 are set in
Phase1BPedTrigger.xml because the parent constructor's defaults (30 / 15) would otherwise move
the bus and trigger point.
"""

import csv
import math
import os

import carla
import py_trees

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime
from srunner.scenarios.bt_parked_with_blindspot_ped import (
    BtParkedWithBlindSpotPed,
    get_location_on_same_road,
    get_value_parameter,
)


PED_ONSET_SPEED = 0.1
DEFAULT_OUT_CSV = (
    "/home/peeradon/autoware/src/av-stack-playground/bt_obstacle_eval/"
    "analysis/determinism/phase1B_ped_trigger/run_00.csv"
)


def speed_of(actor):
    v = actor.get_velocity()
    return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


class OpenLoopPedTrigger(py_trees.behaviour.Behaviour):
    """Drive ego open-loop, trigger walker, and log Phase 1B event fields."""

    def __init__(self, ego, walker, collision_location, trigger_from_center, wmap,
                 ticks, throttle, steer, ped_speed, settle_ticks, out_csv,
                 stop_ego_after_trigger, freeze_ego_after_trigger,
                 name="OpenLoopPedTrigger"):
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
        self._stop_ego_after_trigger = stop_ego_after_trigger
        self._freeze_ego_after_trigger = freeze_ego_after_trigger
        self._ego_frozen = False

        self._i = 0
        self._triggered = False
        self._t_trigger = float("nan")
        self._t_ped_start = float("nan")
        self._csv_file = None
        self._csv = None
        self._walker_dir = None

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
            "x", "y", "yaw", "speed", "road_dist",
        ])
        print(f"[Phase1BPedTrigger] logging to {out_csv}")
        self._walker_dir = self._walker.get_transform().get_forward_vector()
        self._ego.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        self._walker.apply_control(carla.WalkerControl(direction=self._walker_dir, speed=0.0))

    def update(self):
        if self._i < self._settle_ticks:
            self._ego.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
            self._walker.apply_control(carla.WalkerControl(direction=self._walker_dir, speed=0.0))
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

        self._i += 1
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None


class Phase1BPedTrigger(BtParkedWithBlindSpotPed):
    """CARLA-only Phase 1B scenario — reuses the parent blind-spot geometry, drives the ego
    open-loop, and logs per-tick repeatability data via OpenLoopPedTrigger."""

    def __init__(self, world, ego_vehicles, config, randomize=False,
                 debug_mode=False, criteria_enable=True, timeout=120):
        # Phase 1B-specific params — set BEFORE super().__init__(), because the parent
        # constructor runs BasicScenario.__init__ → _initialize_actors / _create_behavior,
        # which read these (same ordering rule as phase2_ros_controller's self._spawn_ped).
        self._ticks = get_value_parameter(config, "ticks", int, 350)
        self._throttle = get_value_parameter(config, "throttle", float, 0.5)
        self._steer = get_value_parameter(config, "steer", float, 0.0)
        self._settle_ticks = get_value_parameter(config, "settle_ticks", int, 20)
        self._walker_filter = get_value_parameter(config, "walker_filter", str, "walker.pedestrian.0001")
        self._out_csv = get_value_parameter(config, "out_csv", str, DEFAULT_OUT_CSV)
        self._stop_ego_after_trigger = (
            get_value_parameter(config, "stop_ego_after_trigger", str, "true").lower() == "true")
        self._freeze_ego_after_trigger = (
            get_value_parameter(config, "freeze_ego_after_trigger", str, "true").lower() == "true")
        self._snap_ego_to_lane_center = (
            get_value_parameter(config, "snap_ego_to_lane_center", str, "true").lower() == "true")
        self._trigger_from_center = None

        # parked_dist / trigger_dist / park_side / yaw_offset / adversary_speed are read by the
        # parent constructor. Phase 1B needs parked_dist=100, trigger_dist=4 (parent defaults are
        # 30 / 15) — set in Phase1BPedTrigger.xml's <other_parameters> so the parent reads them
        # before _initialize_actors places the bus.
        super().__init__(world, ego_vehicles, config, randomize, debug_mode,
                         criteria_enable=False, timeout=timeout)

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
            self._stop_ego_after_trigger,
            self._freeze_ego_after_trigger,
        )

    def _create_test_criteria(self):
        return []

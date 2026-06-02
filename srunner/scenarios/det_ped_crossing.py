#!/usr/bin/env python3
"""
DetPedCrossing — blindspot-geometry pedestrian input determinism scenario.

Purpose: spawn the same parked bus + pedestrian geometry used by
BtParkedWithBlindSpotPed, but without the ego-distance trigger and without Autoware.
ScenarioRunner handles actor lifecycle and places the pedestrian at the real blindspot
start location immediately; walking is controlled externally by run_input_determinism.py via the
CARLA Python API.

Parameters (all optional, set in XML <other_parameters>):
  parked_dist      — ego spawn to parked vehicle rear distance (default 100.0)
  park_side        — left/right shoulder side (default right)
  yaw_offset       — parked vehicle yaw offset (default 0.0)
  adversary_speed  — documented for parity with the main scenario; external script controls speed
  timeout          — scenario timeout in seconds (default 120)
"""

import py_trees
import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenarios.basic_scenario import BasicScenario


class _AlwaysRunning(py_trees.behaviour.Behaviour):
    """Keeps ScenarioRunner alive until killed externally (run_input_determinism.py)."""
    def update(self):
        return py_trees.common.Status.RUNNING


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


class DetPedCrossing(BasicScenario):
    """Determinism-test scenario using the real blindspot parked-bus geometry."""

    def __init__(self, world, ego_vehicles, config, randomize=False,
                 debug_mode=False, criteria_enable=True, timeout=120):

        self._wmap = CarlaDataProvider.get_map()
        self._reference_waypoint = self._wmap.get_waypoint(config.trigger_points[0].location)
        self.timeout = get_value_parameter(config, 'timeout', float, float(timeout))

        self._parked_dist = get_value_parameter(config, 'parked_dist', float, 100.0)
        self._park_side = get_value_parameter(config, 'park_side', str, 'right')
        if self._park_side not in ('left', 'right'):
            raise ValueError(f"'park_side' must be either 'right' or 'left' but '{self._park_side}' was given")
        self._yaw_offset = get_value_parameter(config, 'yaw_offset', float, 0.0)
        self._adversary_speed = get_value_parameter(config, 'adversary_speed', float, 2.0)

        self._bp_attributes = {}

        self._parked_transform = None
        self._ped_transform = None
        self._collision_wp = None
        self._parked_car_half_len = None

        # criteria_enable=False: empty criteria list causes py_trees Parallel to complete
        # immediately (0/0 SUCCESS), which exits the scenario before we can record anything.
        super(DetPedCrossing, self).__init__(
            'DetPedCrossing',
            ego_vehicles,
            config,
            world,
            debug_mode,
            criteria_enable=False,
        )

    # ------------------------------------------------------------------
    def _get_blocker_transform(self, waypoint):
        if waypoint.lane_type == carla.LaneType.Sidewalk:
            new_location = waypoint.transform.location
        else:
            vector = waypoint.transform.get_right_vector()
            if self._park_side == 'left':
                vector *= -1
            offset_location = carla.Location(
                waypoint.lane_width * 0.7 * vector.x,
                waypoint.lane_width * 0.7 * vector.y)
            new_location = waypoint.transform.location + offset_location
        new_location.z += 0.2

        new_rotation = carla.Rotation(
            pitch=waypoint.transform.rotation.pitch,
            yaw=waypoint.transform.rotation.yaw + self._yaw_offset,
            roll=waypoint.transform.rotation.roll)
        return carla.Transform(new_location, new_rotation)

    def _get_walker_transform(self, waypoint):
        new_rotation = waypoint.transform.rotation
        new_rotation.yaw += 270 if self._park_side == 'right' else 90

        if waypoint.lane_type == carla.LaneType.Sidewalk:
            new_location = waypoint.transform.location
        else:
            vector = waypoint.transform.get_right_vector()
            if self._park_side == 'left':
                vector *= -1
            offset_location = carla.Location(waypoint.lane_width * vector.x, waypoint.lane_width * vector.y)
            new_location = waypoint.transform.location + offset_location
        new_location.z += 1.2

        return carla.Transform(new_location, new_rotation)

    # ------------------------------------------------------------------
    def _initialize_actors(self, config):
        # Reset the (persistent) ego to the route start every run, so the determinism
        # approach test (run_input_determinism.py --approach) drives the full approach
        # from the same spawn pose each time instead of teleporting near the bus.
        # The ego is spawned once by the bridge and attached via --waitForEgo, so without
        # this reset run N would start wherever run N-1 stopped.
        start_tf = config.trigger_points[0]
        self.ego_vehicles[0].set_transform(start_tf)
        self.ego_vehicles[0].set_target_velocity(carla.Vector3D(0, 0, 0))
        self.ego_vehicles[0].set_target_angular_velocity(carla.Vector3D(0, 0, 0))

        park_location, _ = get_location_on_same_road(
            self._reference_waypoint, self._parked_dist)
        park_wp = self._wmap.get_waypoint(park_location)
        self._parked_transform = self._get_blocker_transform(park_wp)
        self.parking_slots.append(self._parked_transform.location)

        parked_vehicle = CarlaDataProvider.request_new_actor(
            'vehicle.mitsubishi.fusorosa', self._parked_transform,
            rolename='scenario', attribute_filter=self._bp_attributes)
        if parked_vehicle is None:
            raise ValueError('DetPedCrossing: failed to spawn parked vehicle')

        parked_vehicle.apply_control(carla.VehicleControl(hand_brake=True))
        parked_vehicle.set_simulate_physics(True)
        self._parked_car_half_len = parked_vehicle.bounding_box.extent.x
        self.other_actors.append(parked_vehicle)  # index 0

        walker_dist = parked_vehicle.bounding_box.extent.x + 0.5
        wps = park_wp.next(walker_dist)
        if not wps:
            raise ValueError('DetPedCrossing: could not find walker waypoint')
        walker_wp = wps[0]
        self._collision_wp = walker_wp

        self._ped_transform = self._get_walker_transform(walker_wp)
        self.parking_slots.append(self._ped_transform.location)
        ped = CarlaDataProvider.request_new_actor('walker.*', self._ped_transform)
        if ped is None:
            raise ValueError('DetPedCrossing: failed to spawn pedestrian')

        ped.set_simulate_physics(True)
        self.other_actors.append(ped)  # index 1

        ego_tf = self.ego_vehicles[0].get_transform()
        dist = self._ped_transform.location.distance(ego_tf.location)
        print(f'[DetPedCrossing] parked bus at ({self._parked_transform.location.x:.2f}, '
              f'{self._parked_transform.location.y:.2f}); ped at '
              f'({self._ped_transform.location.x:.2f}, {self._ped_transform.location.y:.2f}); '
              f'dist_from_ego={dist:.2f} m')

    # ------------------------------------------------------------------
    def _create_behavior(self):
        # Stay RUNNING until run_input_determinism.py kills ScenarioRunner.
        # Walking is controlled externally via direct CARLA WalkerControl API.
        return _AlwaysRunning(name='WaitForExternalControl')

    # ------------------------------------------------------------------
    def _create_test_criteria(self):
        return []

    def __del__(self):
        self.remove_all_actors()

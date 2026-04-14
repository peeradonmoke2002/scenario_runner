#!/usr/bin/env python

# MIT License

"""
BtParkedVehicle scenario:

A vehicle is parked (blocking) in the ego's lane on a straight road.
The ego vehicle must detect it via bt_obstacle and generate a shift
path to avoid it — testing the VehicleBranch (IsParkedVehicle) flow
of the BehaviorTree.

Scenario ends when the ego has driven past the obstacle (DriveDistance).
"""

import py_trees
import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (
    ActorTransformSetter,
)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (
    DriveDistance,
    InTriggerDistanceToLocation,
)
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.scenario_helper import get_location_in_distance_from_wp


def get_value_parameter(config, name, p_type, default):
    if name in config.other_parameters:
        return p_type(config.other_parameters[name]['value'])
    return default


class BtParkedVehicle(BasicScenario):
    """
    A parked vehicle blocks the ego's lane ahead.

    bt_obstacle should:
      1. Filter   → IsVehicleType → IsParkedVehicle → CommitVehicleTargets
      2. Decide   → NeedAvoid (avoidable_objects not empty)
      3. Plan     → GenShiftPath → shift left/right around parked car
    """

    timeout = 120

    def __init__(self, world, ego_vehicles, config,
                 randomize=False, debug_mode=False, criteria_enable=True, timeout=120):

        self._wmap = CarlaDataProvider.get_map()
        self._reference_waypoint = self._wmap.get_waypoint(config.trigger_points[0].location)
        self.timeout = timeout

        # How far ahead to spawn the parked car (metres)
        self._spawn_dist = get_value_parameter(config, 'distance', float, 25.0)

        # Which side to park on: 'right' or 'left'
        self._park_side = get_value_parameter(config, 'park_side', str, 'right')

        self._parked_transform = None

        super(BtParkedVehicle, self).__init__(
            "BtParkedVehicle",
            ego_vehicles,
            config,
            world,
            debug_mode,
            criteria_enable=criteria_enable
        )

    # ------------------------------------------------------------------
    def _initialize_actors(self, config):
        """Spawn a car at the road edge so isParkedVehicle() returns true.

        isParkedVehicle checks:
          shiftable_ratio = dist(object→centerline) / dist(centerline→road_boundary)
          → must be > object_check_shiftable_ratio (~0.6)
          → AND abs(to_centerline) >= threshold_distance_object_is_on_center

        So the vehicle must be near the road edge, NOT centred in the lane.
        We offset it by ~80% of half-lane-width toward the road boundary.
        """

        # Walk forward along the road from the trigger point
        location, _ = get_location_in_distance_from_wp(
            self._reference_waypoint, self._spawn_dist)
        waypoint = self._wmap.get_waypoint(location)

        lane_width = waypoint.lane_width  # typically ~3.5 m

        # Offset ~80% toward the road edge so shiftable_ratio > 0.6
        # right_vector points to the right side of the road
        right_vec = waypoint.transform.get_right_vector()
        side_sign = 1.0 if self._park_side == 'right' else -1.0
        edge_offset = side_sign * lane_width * 0.8

        offset = carla.Location(
            edge_offset * right_vec.x,
            edge_offset * right_vec.y,
            0.3
        )
        spawn_location = waypoint.transform.location + offset
        spawn_rotation = waypoint.transform.rotation

        self._parked_transform = carla.Transform(spawn_location, spawn_rotation)

        vehicle = CarlaDataProvider.request_new_actor(
            'vehicle.tesla.model3',
            self._parked_transform,
            rolename='scenario'
        )
        if vehicle is None:
            raise ValueError("BtParkedVehicle: failed to spawn parked vehicle")

        # Handbrake on — it stays parked
        vehicle.apply_control(carla.VehicleControl(hand_brake=True))
        vehicle.set_simulate_physics(True)

        self.other_actors.append(vehicle)

    # ------------------------------------------------------------------
    def _create_behavior(self):
        """
        Tree:
          Parallel (SUCCESS_ON_ONE)
            └─ Sequence
                 ├─ ActorTransformSetter         place parked car — stays forever, never removed
                 ├─ InTriggerDistanceToLocation  wait until ego is alongside parked car (< 10m)
                 └─ DriveDistance(ego, 20m)      ego drives 20m past that point → confirmed passed

        No ActorDestroy — parked car stays in the scene throughout.
        CollisionTest (criteria) catches any collision during avoidance.
        """
        parked_car_location = self._parked_transform.location

        root = py_trees.composites.Parallel(
            name="BtParkedVehicle",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE
        )
        sequence = py_trees.composites.Sequence("ScenarioSequence")

        # 1. Place parked car — stays for the whole scenario
        sequence.add_child(
            ActorTransformSetter(self.other_actors[0], self._parked_transform, name="PlaceCar"))

        # 2. Wait until ego is alongside the parked car (avoidance must happen here)
        sequence.add_child(
            InTriggerDistanceToLocation(
                self.ego_vehicles[0],
                parked_car_location,
                distance=10.0,
                name="EgoAlongsideCar"))

        # 3. Ego drives 20m more → confirmed it fully passed the parked car
        sequence.add_child(
            DriveDistance(self.ego_vehicles[0], 20, name="EgoFullyPassed"))

        root.add_child(sequence)
        return root

    # ------------------------------------------------------------------
    def _create_test_criteria(self):
        return [CollisionTest(self.ego_vehicles[0])]

    def __del__(self):
        self.remove_all_actors()

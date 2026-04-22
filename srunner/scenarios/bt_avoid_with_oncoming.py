#!/usr/bin/python3

# MIT License

"""
BtAvoidWithOncoming scenario:

Ego approaches a parked car (in ego lane, near road edge).
bt_obstacle starts generating a shift path to avoid it.
While ego is mid-shift, an oncoming vehicle enters the avoidance lane
from ahead and drives toward ego.

Expected bt_obstacle behavior (Fix 14):
  1. PerceptionFilter → IsParkedVehicle → avoidable_objects = [parked_car]
  2. NeedAvoid → SUCCESS → GenShiftPath → ego begins shifting left
  3. CheckPathSafety: isSafePath detects oncoming car → safe=false
     → needs_stop=true → FAILURE
  4. NeedStop → SUCCESS → StopBehavior inserts wait point on shifted path
     → ego halts before leaving original lane
  5. Oncoming car passes → isSafePath=true → CheckPathSafety SUCCESS
  6. Plan resumes → ego completes avoidance around parked car

Scenario ends when ego has driven past the parked car (DriveDistance).
"""

import py_trees
import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (
    ActorTransformSetter,
    WaypointFollower,
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


class BtAvoidWithOncoming(BasicScenario):
    """
    Parked car in ego lane + oncoming vehicle enters avoidance lane mid-shift.

    Tests Fix 14: CheckPathSafety RSS fail mid-maneuver → yield-by-stopping,
    then resumes avoidance when oncoming clears.

    Parameters (via XML other_parameters):
      parked_dist     — distance ahead to spawn parked car (m), default 30
      park_side       — 'right' or 'left' edge of ego lane, default 'right'
      yaw_offset      — extra yaw rotation of parked car (degrees), default 15
                        positive = nose points into ego lane → forces larger lateral shift
      oncoming_dist   — distance ahead to spawn oncoming car (m), default 50
      oncoming_speed  — speed of oncoming car toward ego (m/s), default 8.0
      trigger_dist    — how close ego must be to parked car before oncoming starts (m), default 18
    """

    timeout = 150

    def __init__(self, world, ego_vehicles, config,
                 randomize=False, debug_mode=False, criteria_enable=True, timeout=150):

        self._wmap = CarlaDataProvider.get_map()
        self._reference_waypoint = self._wmap.get_waypoint(config.trigger_points[0].location)
        self.timeout = timeout

        self._parked_dist    = get_value_parameter(config, 'parked_dist',    float, 30.0)
        self._park_side      = get_value_parameter(config, 'park_side',      str,   'right')
        self._yaw_offset     = get_value_parameter(config, 'yaw_offset',     float, 15.0)
        self._oncoming_dist  = get_value_parameter(config, 'oncoming_dist',  float, 50.0)
        self._oncoming_speed = get_value_parameter(config, 'oncoming_speed', float, 8.0)
        self._trigger_dist   = get_value_parameter(config, 'trigger_dist',   float, 18.0)

        self._parked_transform  = None
        self._oncoming_transform = None

        super(BtAvoidWithOncoming, self).__init__(
            "BtAvoidWithOncoming",
            ego_vehicles,
            config,
            world,
            debug_mode,
            criteria_enable=criteria_enable
        )

    # ------------------------------------------------------------------
    def _initialize_actors(self, config):

        # ---- 1. Parked car — near road edge in ego lane ----
        park_location, _ = get_location_in_distance_from_wp(
            self._reference_waypoint, self._parked_dist)
        park_wp = self._wmap.get_waypoint(park_location)

        lane_width = park_wp.lane_width
        right_vec  = park_wp.transform.get_right_vector()
        side_sign  = 1.0 if self._park_side == 'right' else -1.0
        spawn_rot  = carla.Rotation(
            pitch=park_wp.transform.rotation.pitch,
            yaw=park_wp.transform.rotation.yaw + self._yaw_offset,
            roll=park_wp.transform.rotation.roll
        )

        parked_vehicle = None
        for lateral_ratio in [0.8, 0.6, 0.4]:
            edge_offset = side_sign * lane_width * lateral_ratio
            base_loc = park_wp.transform.location + carla.Location(
                edge_offset * right_vec.x,
                edge_offset * right_vec.y,
                0.0
            )
            for z_offset in [0.5, 1.0, 1.5]:
                self._parked_transform = carla.Transform(
                    carla.Location(base_loc.x, base_loc.y, park_wp.transform.location.z + z_offset),
                    spawn_rot
                )
                parked_vehicle = CarlaDataProvider.request_new_actor(
                    'vehicle.tesla.model3',
                    self._parked_transform,
                    rolename='scenario'
                )
                if parked_vehicle is not None:
                    break
            if parked_vehicle is not None:
                break

        if parked_vehicle is None:
            raise ValueError("BtAvoidWithOncoming: failed to spawn parked vehicle")

        parked_vehicle.apply_control(carla.VehicleControl(hand_brake=True))
        parked_vehicle.set_simulate_physics(True)
        self.other_actors.append(parked_vehicle)  # index 0

        # ---- 2. Oncoming car — opposite lane, further ahead, facing toward ego ----
        # Get waypoint in ego's lane at oncoming_dist ahead
        oncoming_location, _ = get_location_in_distance_from_wp(
            self._reference_waypoint, self._oncoming_dist)
        oncoming_wp = self._wmap.get_waypoint(oncoming_location)

        # Move to the left lane (oncoming direction in two-way road)
        left_wp = oncoming_wp.get_left_lane()
        if left_wp is None:
            # Fallback: use same lane center if no left lane found
            left_wp = oncoming_wp

        oncoming_vehicle = None
        for z_offset in [0.5, 1.0, 1.5]:
            self._oncoming_transform = carla.Transform(
                carla.Location(
                    left_wp.transform.location.x,
                    left_wp.transform.location.y,
                    left_wp.transform.location.z + z_offset
                ),
                left_wp.transform.rotation  # already faces toward ego in oncoming lane
            )
            oncoming_vehicle = CarlaDataProvider.request_new_actor(
                'vehicle.tesla.model3',
                self._oncoming_transform,
                rolename='scenario'
            )
            if oncoming_vehicle is not None:
                break

        if oncoming_vehicle is None:
            raise ValueError("BtAvoidWithOncoming: failed to spawn oncoming vehicle")

        oncoming_vehicle.set_simulate_physics(True)
        self.other_actors.append(oncoming_vehicle)  # index 1

    # ------------------------------------------------------------------
    def _create_behavior(self):
        """
        Parallel (SuccessOnOne)           ← killed as soon as sequence finishes
          Sequence
            PlaceParked
            PlaceOncoming
            InTriggerDistanceToLocation(ego, parked_loc, trigger_dist)
              ← ego is approaching parked car, avoidance shift about to start
            Parallel (SuccessOnOne)       ← inner: success once ego fully past
              Sequence
                InTriggerDistanceToLocation(ego, parked_loc, 5m)
                DriveDistance(ego, 25m)   ← ego confirmed past parked car
              WaypointFollower(oncoming)  ← drives toward ego until killed
        """
        parked_actor   = self.other_actors[0]
        oncoming_actor = self.other_actors[1]
        parked_loc     = self._parked_transform.location

        root = py_trees.composites.Parallel(
            name="BtAvoidWithOncoming",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE
        )
        seq = py_trees.composites.Sequence("ScenarioSequence", memory=True)

        # Place both actors at their spawn transforms
        seq.add_child(ActorTransformSetter(parked_actor,   self._parked_transform,   name="PlaceParked"))
        seq.add_child(ActorTransformSetter(oncoming_actor, self._oncoming_transform, name="PlaceOncoming"))

        # Wait until ego is close enough to parked car → avoidance shift starting
        seq.add_child(InTriggerDistanceToLocation(
            self.ego_vehicles[0],
            parked_loc,
            distance=self._trigger_dist,
            name="EgoApproachingParked"
        ))

        # Inner parallel: oncoming drives while we wait for ego to pass
        inner = py_trees.composites.Parallel(
            name="OncomingAndWait",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE
        )

        # Success branch: ego gets alongside then drives past
        success_seq = py_trees.composites.Sequence("EgoPassSequence", memory=True)
        success_seq.add_child(InTriggerDistanceToLocation(
            self.ego_vehicles[0],
            parked_loc,
            distance=5.0,
            name="EgoAlongsideCar"
        ))
        success_seq.add_child(DriveDistance(
            self.ego_vehicles[0], 25, name="EgoFullyPassed"
        ))

        # Oncoming car drives in its lane toward ego (killed by SUCCESS_ON_ONE above)
        oncoming_follow = WaypointFollower(
            oncoming_actor,
            target_speed=self._oncoming_speed,
            name="OncomingDrives"
        )

        inner.add_child(success_seq)
        inner.add_child(oncoming_follow)

        seq.add_child(inner)
        root.add_child(seq)
        return root

    # ------------------------------------------------------------------
    def _create_test_criteria(self):
        return [CollisionTest(self.ego_vehicles[0])]

    def __del__(self):
        self.remove_all_actors()

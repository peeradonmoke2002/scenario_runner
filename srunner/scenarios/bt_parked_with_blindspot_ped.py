#!/usr/bin/env python3

# MIT License

"""
BtParkedWithBlindSpotPed scenario:

Ego approaches a parked car (right side of ego lane, near road edge).
bt_obstacle begins generating a left-shift avoidance path.
When ego is mid-approach (trigger_dist from parked car), a pedestrian
steps out from behind the parked car into the avoidance path — the classic
"blind spot" hazard created by the parked vehicle.

Expected bt_obstacle behavior:
  1. PerceptionFilter → IsParkedVehicle → avoidable_objects = [parked_car]
  2. NeedAvoid → SUCCESS → GenShiftPath → ego begins shifting left
  3. Pedestrian steps out from blind spot (hidden behind parked car) into avoidance path:
       NonVehicleBranch: IsNonVehicleType → IsNotOnEdgeLane → MarkNonVehicleStop2BB
       → stop_target_object set; needs_stop = true
  4. NeedAvoid: needs_stop=true AND ped at same/closer distance as parked car
       → Fix 5 does NOT apply → FAILURE
     NeedStop → SUCCESS → StopBehavior → ego halts
  5. Pedestrian walks through and clears the avoidance path
       → stop_target_object empty; needs_stop = false
  6. NeedAvoid resumes → AvoidBehavior → Plan → ego completes avoidance around parked car ✓

Parameters (via XML other_parameters):
  parked_dist   — distance ahead to spawn parked car (m), default 30
  park_side     — 'right' or 'left' edge of ego lane, default 'right'
  yaw_offset    — extra yaw rotation of parked car (degrees), default -15
                  negative = nose points into ego lane
  ped_speed     — pedestrian walking speed (m/s), default 1.2
  trigger_drive — distance ego drives from start before ped begins walking (m), default 3
                  small value = ped starts early → perception sees ped before avoidance begins
  pass_dist     — distance ego drives past the parked car to mark scenario done (m), default 30
"""

import py_trees
import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (
    ActorTransformSetter,
    KeepVelocity,
)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (
    DriveDistance,
)
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.scenario_helper import get_location_in_distance_from_wp


def get_value_parameter(config, name, p_type, default):
    if name in config.other_parameters:
        return p_type(config.other_parameters[name]['value'])
    return default


class BtParkedWithBlindSpotPed(BasicScenario):
    """
    Parked car in ego lane + pedestrian steps out from behind it during avoidance.

    Tests the blind-spot pedestrian hazard:
      - Parked car triggers avoidable_objects (IsParkedVehicle path)
      - Pedestrian appears from behind parked car into the avoidance lane
      - NonVehicleBranch → MarkNonVehicleStop2BB → stop_target_object; needs_stop=true
      - NeedStop → StopBehavior → ego yields until pedestrian clears
      - Avoidance then completes around parked car

    Timing design (ego ~8 m/s, parked_dist=30, trigger_drive=3):
      ego drives 3m from start → t≈0.4s → ped starts walking (1.2 m/s)
      ped walks ~1.8m to lane center → visible to sensors at t≈1.9s
      ego is ~14m from parked car when ped detected → PerceptionFilter sees ped
      BEFORE avoidance starts → NeedStop fires → ego stops with room to spare
      ped clears (~5s) → NeedAvoid resumes → ego avoids parked car
    """

    timeout = 150

    def __init__(self, world, ego_vehicles, config,
                 randomize=False, debug_mode=False, criteria_enable=True, timeout=150):

        self._wmap = CarlaDataProvider.get_map()
        self._reference_waypoint = self._wmap.get_waypoint(config.trigger_points[0].location)
        self.timeout = timeout

        self._parked_dist  = get_value_parameter(config, 'parked_dist',  float, 30.0)
        self._park_side    = get_value_parameter(config, 'park_side',    str,   'right')
        self._yaw_offset   = get_value_parameter(config, 'yaw_offset',   float, -15.0)
        self._ped_speed      = get_value_parameter(config, 'ped_speed',     float, 1.2)
        self._trigger_drive  = get_value_parameter(config, 'trigger_drive', float, 3.0)
        self._pass_dist      = get_value_parameter(config, 'pass_dist',     float, 30.0)

        self._parked_transform = None
        self._ped_transform    = None

        super(BtParkedWithBlindSpotPed, self).__init__(
            "BtParkedWithBlindSpotPed",
            ego_vehicles,
            config,
            world,
            debug_mode,
            criteria_enable=criteria_enable
        )

    # ------------------------------------------------------------------
    def _initialize_actors(self, config):
        """
        Follows the cut_in_with_static_vehicle.py pattern:
          - All actors spawned at final position, physics=False, then moved to z=-500
          - ActorTransformSetter in _create_behavior places them (parked stays kinematic)
        This prevents the physics explosion that occurred when the pedestrian teleported
        next to a physics-enabled parked car.
        """

        # ---- 1. Parked car — kinematic static obstacle at road edge ----
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
            self._parked_transform = carla.Transform(
                carla.Location(base_loc.x, base_loc.y,
                               park_wp.transform.location.z + 0.2),
                spawn_rot
            )
            parked_vehicle = CarlaDataProvider.request_new_actor(
                'vehicle.tesla.model3',
                self._parked_transform,
                rolename='scenario'
            )
            if parked_vehicle is not None:
                break

        if parked_vehicle is None:
            raise ValueError("BtParkedWithBlindSpotPed: failed to spawn parked vehicle")

        parked_vehicle.apply_control(carla.VehicleControl(hand_brake=True))
        parked_vehicle.set_simulate_physics(True)
        self.other_actors.append(parked_vehicle)  # index 0

        # ---- 2. Pedestrian — 3m past parked car, 1.2x lane_width lateral ----
        # Placed 3m FURTHER from ego than the parked car (behind it from ego's view)
        # so the parked car partially occludes the pedestrian until they step out.
        # At 1.2x lane_width the pedestrian is well outside the parked car's bounding
        # box, so there is zero bounding-box overlap when ActorTransformSetter places them.
        ped_location, _ = get_location_in_distance_from_wp(
            self._reference_waypoint, self._parked_dist + 3.0)
        ped_wp = self._wmap.get_waypoint(ped_location)

        ped_lateral = side_sign * lane_width * 1.2
        ped_base_loc = ped_wp.transform.location + carla.Location(
            ped_lateral * right_vec.x,
            ped_lateral * right_vec.y,
            0.0
        )

        # Walker faces perpendicular to road so KeepVelocity walks them straight across.
        ped_yaw = park_wp.transform.rotation.yaw - (90.0 * side_sign)
        ped_rot = carla.Rotation(pitch=0.0, yaw=ped_yaw, roll=0.0)

        self._ped_transform = carla.Transform(
            carla.Location(ped_base_loc.x, ped_base_loc.y,
                           ped_wp.transform.location.z + 1.0),
            ped_rot
        )

        # Spawn walker using pedestrian_crossing.py pattern:
        #   spawn near ego (z-50), then set_location deeper (z-100).
        # z-500 exceeded CARLA world bounds → set_transform() silently failed.
        # z-100 is reliably within bounds and matches the _replace_walker() approach.
        ref_loc = self._reference_waypoint.transform.location
        spawn_underground = carla.Transform(
            carla.Location(ref_loc.x, ref_loc.y, ref_loc.z - 50.0),
            ped_rot
        )
        pedestrian = CarlaDataProvider.request_new_actor(
            'walker.pedestrian.0001', spawn_underground, rolename='scenario')

        if pedestrian is None:
            raise ValueError("BtParkedWithBlindSpotPed: failed to spawn pedestrian")

        pedestrian.set_simulate_physics(True)
        pedestrian.set_location(carla.Location(ref_loc.x, ref_loc.y, ref_loc.z - 100.0))
        self.other_actors.append(pedestrian)  # index 1

    # ------------------------------------------------------------------
    def _create_behavior(self):
        """
        Parallel (SuccessOnOne)
          Sequence
            PlacePedestrian  (ActorTransformSetter — polls until server confirms transform)
              ← ped is at 1.2x lane_width (road edge) so bt_obstacle IsNotOnEdgeLane FAILS
                 → ped filtered out while stationary; only detected when walking into lane
            DriveDistance(ego, trigger_drive=3m)
              ← ped starts walking almost immediately after ego moves; gives ~14m runway
                 before parked car so PerceptionFilter detects ped BEFORE avoidance starts
            Parallel (SuccessOnOne)
              Sequence
                InTriggerDistanceToLocation(ego, parked_loc, 5m)
                DriveDistance(ego, 25m)   ← ego confirmed past parked car
              KeepVelocity(ped, ped_speed, 20s/10m)
                ← ped walks perpendicular across avoidance path (killed when ego passes)
        """
        root = py_trees.composites.Parallel(
            name="BtParkedWithBlindSpotPed",
            policy=py_trees.common.ParallelPolicy.SuccessOnAll()
        )
        seq = py_trees.composites.Sequence("ScenarioSequence", memory=True)

        seq.add_child(ActorTransformSetter(
            self.other_actors[1], self._ped_transform, name="PlacePedestrian"))

        seq.add_child(DriveDistance(
            self.ego_vehicles[0], self._trigger_drive, name="EgoStartedMoving"
        ))

        # Ped walks to other shoulder (10 m) then stops — seq returns SUCCESS.
        # RunForever keeps root RUNNING so scenario never ends automatically.
        seq.add_child(KeepVelocity(
            self.other_actors[1],
            self._ped_speed,
            False,
            duration=float("inf"),
            name="PedWalksAcross"
        ))

        root.add_child(seq)
        root.add_child(py_trees.behaviours.Running(name="RunForever"))
        return root

    # ------------------------------------------------------------------
    def _create_test_criteria(self):
        return [CollisionTest(self.ego_vehicles[0])]

    # def __del__(self):
    #     self.remove_all_actors()
    

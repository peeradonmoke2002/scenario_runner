#!/usr/bin/python3

import csv
import os
import time

import py_trees
import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (
    ActorDestroy,
    ActorTransformSetter,
    KeepVelocity,
    MovePedestrianWithEgo,
)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (
    DriveDistance,
)
from srunner.scenarios.bt_ped_response import (
    PED_RESPONSES,
    CrossingPedResponse,
    get_longitudinal_bbox_extents,
)


class InLaneTriggerDistance(py_trees.behaviour.Behaviour):
    """
    Trigger when ego's lane-projected distance to a target drops below threshold.
    Projects ego onto the nearest lane waypoint before measuring distance,
    eliminating lateral variation caused by avoidance shifts.

    Debug: logs every tick to /tmp/bt_trigger_debug.csv
           columns: wall_t, road_dist, ego_speed, fired
    """
    DEBUG_DIR = "/tmp/bt_trigger_debug"

    def __init__(self, ego, target_location, distance, wmap, name="InLaneTriggerDistance"):
        super().__init__(name)
        self._ego = ego
        self._target_location = target_location
        self._distance = distance
        self._wmap = wmap
        self._t0_wall = None
        self._fired = False
        # one CSV per run — run_id comes from /tmp/bt_run_id.txt (written by run_blindspot.sh)
        os.makedirs(self.DEBUG_DIR, exist_ok=True)
        try:
            with open("/tmp/bt_run_id.txt") as f:
                run_id = f.read().strip()
        except Exception:
            run_id = str(int(time.time()))
        fname = os.path.join(self.DEBUG_DIR, f"trigger_{run_id}.csv")
        self._csv_file = open(fname, "w", newline="")
        self._csv = csv.writer(self._csv_file)
        self._csv.writerow(["wall_t", "road_dist", "ego_speed", "fired"])
        print(f"[TriggerDebug] logging to {fname}")

    def update(self):
        now = time.monotonic()
        if self._t0_wall is None:
            self._t0_wall = now
        rel_t = now - self._t0_wall

        ego_wp    = self._wmap.get_waypoint(self._ego.get_location())
        road_dist = ego_wp.transform.location.distance(self._target_location)
        ego_v     = self._ego.get_velocity()
        ego_speed = (ego_v.x**2 + ego_v.y**2 + ego_v.z**2) ** 0.5

        if road_dist <= self._distance and not self._fired:
            self._fired = True
            self._csv.writerow([round(rel_t, 4), round(road_dist, 4), round(ego_speed, 4), 1])
            self._csv_file.flush()
            print(f"[TriggerFired] t={rel_t:.3f}s  road_dist={road_dist:.3f}m  ego_speed={ego_speed:.3f} m/s")
            return py_trees.common.Status.SUCCESS

        self._csv.writerow([round(rel_t, 4), round(road_dist, 4), round(ego_speed, 4), 0])
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        self._csv_file.close()
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.background_manager import LeaveSpaceInFront, LeaveCrossingSpace


def get_value_parameter(config, name, p_type, default):
    if name in config.other_parameters:
        return p_type(config.other_parameters[name]['value'])
    return default


def get_location_on_same_road(start_wp, distance, step=1.0):
    """Follow waypoints up to distance but stop if road_id changes (junction crossed)."""
    traveled = 0
    wp = start_wp
    while traveled < distance:
        nexts = wp.next(step)
        if not nexts:
            break
        wp_new = nexts[0]  # always take straight-ahead (first option)
        if wp_new.road_id != start_wp.road_id:
            print(f"[DEBUG] road_id changed {start_wp.road_id}→{wp_new.road_id} at traveled={traveled:.1f}m, stopping")
            break
        traveled += wp_new.transform.location.distance(wp.transform.location)
        wp = wp_new
    return wp.transform.location, traveled


class BtParkedWithBlindSpotPed(BasicScenario):

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=150):

        self._wmap = CarlaDataProvider.get_map()
        self._reference_waypoint = self._wmap.get_waypoint(config.trigger_points[0].location)

        self.timeout = timeout

        self._parked_dist      = get_value_parameter(config, 'parked_dist',      float, 30.0)
        self._park_side        = get_value_parameter(config, 'park_side',         str,   'right')
        if self._park_side not in ('left', 'right'):
            raise ValueError(f"'park_side' must be either 'right' or 'left' but '{self._park_side}' was given")
        self._yaw_offset       = get_value_parameter(config, 'yaw_offset',        float, 0.0)
        self._adversary_speed  = get_value_parameter(config, 'adversary_speed',    float, 2.0)

        self._trigger_dist = get_value_parameter(config, 'trigger_dist', float, 15.0)

        # ped_offset = gap (m) past the parked car's FRONT bumper where the pedestrian
        # crosses. Default 0.5 reproduces the previous hardcoded placement (extent.x + 0.5),
        # so existing runs are unchanged; larger values push the crossing point further
        # downroad (into the ego's path later).
        self._ped_offset = get_value_parameter(config, 'ped_offset', float, 0.5)

        # Lateral placement from the driving-lane centerline. 0.5 is the lane edge; keep this
        # above 0.5 so the pedestrian starts close to the road, but not on the road.
        self._ped_lateral_offset_ratio = get_value_parameter(
            config, 'ped_lateral_offset_ratio', float, 0.99)
        if self._ped_lateral_offset_ratio <= 0.5:
            raise ValueError(
                "'ped_lateral_offset_ratio' must be > 0.5 to keep the pedestrian off the road")

        # Responding pedestrian (Scenario 1, Experiment 3). 'none' (default) keeps the paper
        # behaviour: constant-speed KeepVelocity across the road. Any other value uses the
        # Scenario 2 CrossingPedResponse rules (bt_ped_response.py) with the decision point at
        # decision_offset right of the lane centre: 1.0 m by default, about where the pedestrian
        # first comes out from behind the bus (first LiDAR detection in the paper runs was
        # 1.0-1.2 m right of the centre). A negative value puts the decision point left of the
        # centre, e.g. -1.0 = the middle of the ego's bus-avoidance path.
        self._ped_response = str(config.other_parameters.get(
            'ped_response', {'value': 'none'})['value']).strip().lower()
        if self._ped_response not in PED_RESPONSES:
            raise ValueError(
                f"BtParkedWithBlindSpotPed: ped_response must be one of {PED_RESPONSES}, "
                f"got '{self._ped_response}'")
        self._gap_accept_m    = get_value_parameter(config, 'gap_accept_m',    float, 33.0)
        self._freeze_dist     = get_value_parameter(config, 'freeze_dist',     float, 25.0)
        self._freeze_hold     = get_value_parameter(config, 'freeze_hold',     float, 0.0)
        self._hurry_speed     = get_value_parameter(config, 'hurry_speed',     float, 2.0)
        self._startup_time    = get_value_parameter(config, 'startup_time',    float, 2.0)
        self._yield_timeout   = get_value_parameter(config, 'yield_timeout',   float, 5.0)
        self._decision_offset = get_value_parameter(config, 'decision_offset', float, 1.0)
        if not 0.0 < self._freeze_dist <= self._gap_accept_m:
            raise ValueError("BtParkedWithBlindSpotPed: need 0 < freeze_dist <= gap_accept_m")
        if min(self._hurry_speed, self._yield_timeout) <= 0.0 \
                or min(self._startup_time, self._freeze_hold) < 0.0:
            raise ValueError(
                "BtParkedWithBlindSpotPed: hurry_speed and yield_timeout must be > 0; "
                "startup_time and freeze_hold must be >= 0")

        self._bp_attributes    = {}

        self._parked_transform  = None
        self._ped_transform     = None
        self._collision_wp      = None
        self._parked_car_half_len = None   # actual bounding box, set in _initialize_actors

        super(BtParkedWithBlindSpotPed, self).__init__(
            "BtParkedWithBlindSpotPed",
            ego_vehicles,
            config,
            world,
            debug_mode,
            criteria_enable=criteria_enable
        )

    # ------------------------------------------------------------------
    def _get_blocker_transform(self, waypoint):
        if waypoint.lane_type == carla.LaneType.Sidewalk:
            new_location = waypoint.transform.location
        else:
            vector = waypoint.transform.get_right_vector()
            if self._park_side == 'left':
                vector *= -1
            # 0.7 * lane_width pushes the bus to the road shoulder, reducing overhang into ego lane.
            # 0.5 was placing the bus with ~0.29m into ego lane → large avoiding_shift → jerk limit exceeded
            # → SLGen produced 0 lines at highway speed → ego had to stop before avoidance could start.
            offset_location = carla.Location(
                waypoint.lane_width * 0.7 * vector.x,
                waypoint.lane_width * 0.7 * vector.y)
            new_location = waypoint.transform.location + offset_location
        new_location.z += 0.2

        new_rotation = carla.Rotation(
            pitch=waypoint.transform.rotation.pitch,
            yaw=waypoint.transform.rotation.yaw + self._yaw_offset,
            roll=waypoint.transform.rotation.roll
        )
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
            offset_location = carla.Location(
                waypoint.lane_width * self._ped_lateral_offset_ratio * vector.x,
                waypoint.lane_width * self._ped_lateral_offset_ratio * vector.y)
            new_location = waypoint.transform.location + offset_location
        new_location.z += 1.2

        return carla.Transform(new_location, new_rotation)

    # ------------------------------------------------------------------
    def _initialize_actors(self, _config):

        # ---- 1. Parked car ----
        park_location, _ = get_location_on_same_road(
            self._reference_waypoint, self._parked_dist)
        park_wp = self._wmap.get_waypoint(park_location)
        self._parked_transform = self._get_blocker_transform(park_wp)
        self.parking_slots.append(self._parked_transform.location)

        with open('/tmp/bt_park_actual_x.txt', 'w') as f:
            f.write(f"{self._parked_transform.location.x:.3f}")

        parked_vehicle = CarlaDataProvider.request_new_actor(
            'vehicle.mitsubishi.fusorosa', self._parked_transform,
            rolename='scenario', attribute_filter=self._bp_attributes)
        if parked_vehicle is None:
            raise ValueError("BtParkedWithBlindSpotPed: failed to spawn parked vehicle")

        parked_vehicle.apply_control(carla.VehicleControl(hand_brake=True))
        parked_vehicle.set_simulate_physics(True)
        self._parked_car_half_len = parked_vehicle.bounding_box.extent.x
        self.other_actors.append(parked_vehicle)  # index 0

        # ---- 2. Pedestrian — positioned using actual bounding box of parked car ----
        # extent.x reaches the bus front; + ped_offset is the gap past it (see __init__).
        walker_dist = parked_vehicle.bounding_box.extent.x + self._ped_offset
        wps = park_wp.next(walker_dist)
        if not wps:
            raise ValueError("BtParkedWithBlindSpotPed: couldn't find walker waypoint")
        walker_wp = wps[0]
        self._collision_wp = walker_wp

        self._ped_transform = self._get_walker_transform(walker_wp)
        self.parking_slots.append(self._ped_transform.location)

        pedestrian = CarlaDataProvider.request_new_actor('walker.*', self._ped_transform)
        if pedestrian is None:
            raise ValueError("BtParkedWithBlindSpotPed: failed to spawn pedestrian")

        pedestrian.set_location(self._ped_transform.location + carla.Location(z=-200))
        pedestrian = self._replace_walker(pedestrian)

        self.other_actors.append(pedestrian)  # index 1

    # ------------------------------------------------------------------
    def _create_behavior(self):

        sequence = py_trees.composites.Sequence("BtParkedWithBlindSpotPed", memory=True)

        # parked_dist = ego front → parked car REAR
        # trigger_dist = ego front → parked car REAR at trigger time
        ego_half_len = self.ego_vehicles[0].bounding_box.extent.x
        pchl = self._parked_car_half_len  # actual parked car half-length from bbox

        if self.route_mode:
            total_dist = self._parked_dist + ego_half_len + 2 * pchl + 15
            sequence.add_child(LeaveSpaceInFront(total_dist))

        sequence.add_child(ActorTransformSetter(
            self.other_actors[1], self._ped_transform, True, name="PlacePedestrian"))

        collision_location = self._collision_wp.transform.location

        # trigger fires when ego front is trigger_dist from parked car rear.
        # ego_center → ped = trigger_dist + ego_half_len + (bus_rear→ped). The ped sits at
        # bus_front + ped_offset = bus_rear + 2*pchl + ped_offset, so the last term is
        # 2*pchl + ped_offset. (Was hardcoded +0.5 = the old fixed ped_offset; must track it,
        # else changing ped_offset shifts where the ego is when the ped fires.)
        trigger_from_center = self._trigger_dist + ego_half_len + 2 * pchl + self._ped_offset

        sequence.add_child(InLaneTriggerDistance(
            self.ego_vehicles[0], collision_location, trigger_from_center,
            self._wmap, name="TriggerAdversaryStart"))

        if self.route_mode:
            sequence.add_child(LeaveCrossingSpace(self._collision_wp))

        # 3x lane_width guarantees the walker clears from shoulder to opposite shoulder
        crossing_dist = self._collision_wp.lane_width * 3.0
        if self._ped_response == 'none':
            crossing_duration = crossing_dist / self._adversary_speed
            sequence.add_child(KeepVelocity(
                self.other_actors[1],
                self._adversary_speed,
                duration=crossing_duration,
                distance=crossing_dist,
                name="PedWalksAcross"
            ))
        else:
            # Same walk and crossing length as above, with the Scenario 2 response at the
            # decision point. d is measured along the road from the ego front to the near edge
            # of the pedestrian box, as in Scenario 2 (not to the bus rear like trigger_dist).
            ped = self.other_actors[1]
            road_forward = self._collision_wp.transform.get_forward_vector()
            ped_upstream, ped_downstream = get_longitudinal_bbox_extents(
                ped, self._ped_transform, road_forward)
            start_offset = self._collision_wp.lane_width * self._ped_lateral_offset_ratio
            decision_walk = start_offset - self._decision_offset
            if not 0.0 < decision_walk < crossing_dist:
                raise ValueError(
                    "BtParkedWithBlindSpotPed: the decision point must lie on the crossing, "
                    f"between the spawn point ({start_offset:.2f} m right of the lane centre) "
                    f"and its end ({crossing_dist - start_offset:.2f} m left); "
                    f"got decision_offset {self._decision_offset:.2f} m")
            sequence.add_child(CrossingPedResponse(
                ped, self.ego_vehicles[0], collision_location, road_forward,
                self._ped_transform.get_forward_vector(),
                {'response': self._ped_response, 'walk_speed': self._adversary_speed,
                 'hurry_speed': self._hurry_speed, 'gap_accept_m': self._gap_accept_m,
                 'freeze_dist': self._freeze_dist, 'freeze_hold': self._freeze_hold,
                 'startup_time': self._startup_time,
                 'yield_timeout': self._yield_timeout,
                 'decision_walk': decision_walk, 'cross_dist': crossing_dist,
                 'ego_half_len': ego_half_len, 'ped_upstream': ped_upstream,
                 'ped_downstream': ped_downstream},
                name="PedCrossesWithResponse"))

        sequence.add_child(ActorDestroy(self.other_actors[1], name="DestroyPedestrian"))
        # Disable physics on parked car so ego can pass without collision/sensor crash
        sequence.add_child(ActorTransformSetter(
            self.other_actors[0], self._parked_transform, physics=False, name="DisableParkedCarPhysics"))
        sequence.add_child(DriveDistance(self.ego_vehicles[0], self._parked_dist + ego_half_len + 2 * pchl + 10, name="EndCondition"))
        sequence.add_child(ActorDestroy(self.other_actors[0], name="DestroyParkedCar"))

        return sequence

    # ------------------------------------------------------------------
    def _create_test_criteria(self):
        if self.route_mode:
            return []
        return [CollisionTest(self.ego_vehicles[0])]

    def __del__(self):
        try:
            self.remove_all_actors()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Pedestrian large-map dormancy workaround (same pattern as ParkingCrossingPedestrian)
    def _replace_walker(self, walker):
        type_id = walker.type_id
        CarlaDataProvider.remove_actor_by_id(walker.id)  # removes from pool AND destroys
        spawn_transform = self.ego_vehicles[0].get_transform()
        spawn_transform.location.z -= 50
        walker = CarlaDataProvider.request_new_actor(type_id, spawn_transform)
        if not walker:
            raise ValueError("BtParkedWithBlindSpotPed: couldn't spawn walker substitute")
        walker.set_simulate_physics(False)
        walker.set_location(spawn_transform.location + carla.Location(z=-50))
        return walker

    def _setup_scenario_trigger(self, config):
        trigger_tree = super()._setup_scenario_trigger(config)

        if not self.route_mode:
            return trigger_tree

        # Keep pedestrian active as ego approaches (large-map fix)
        parallel = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SuccessOnOne(), name="ScenarioTrigger")
        parallel.add_child(MovePedestrianWithEgo(self.ego_vehicles[0], self.other_actors[1], 100))
        parallel.add_child(trigger_tree)
        return parallel

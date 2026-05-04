#!/usr/bin/python3

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


class InLaneTriggerDistance(py_trees.behaviour.Behaviour):
    """
    Trigger when ego's lane-projected distance to a target drops below threshold.
    Projects ego onto the nearest lane waypoint before measuring distance,
    eliminating lateral variation caused by avoidance shifts.
    """
    def __init__(self, ego, target_location, distance, wmap, name="InLaneTriggerDistance"):
        super().__init__(name)
        self._ego = ego
        self._target_location = target_location
        self._distance = distance
        self._wmap = wmap

    def update(self):
        ego_wp = self._wmap.get_waypoint(self._ego.get_location())
        road_dist = ego_wp.transform.location.distance(self._target_location)
        if road_dist <= self._distance:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.background_manager import LeaveSpaceInFront, LeaveCrossingSpace
from srunner.tools.scenario_helper import get_location_in_distance_from_wp


def get_value_parameter(config, name, p_type, default):
    if name in config.other_parameters:
        return p_type(config.other_parameters[name]['value'])
    return default


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

        self._bp_attributes    = {'base_type': 'car', 'generation': 2}

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
            # 0.5 * lane_width places the car at the lane edge (blocking ego's path)
            # lane_width * 1.0 would place it outside the lane (shoulder) like ParkingCrossingPedestrian
            offset_location = carla.Location(
                waypoint.lane_width * 0.5 * vector.x,
                waypoint.lane_width * 0.5 * vector.y)
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
            offset_location = carla.Location(waypoint.lane_width * vector.x, waypoint.lane_width * vector.y)
            new_location = waypoint.transform.location + offset_location
        new_location.z += 1.2

        return carla.Transform(new_location, new_rotation)

    # ------------------------------------------------------------------
    def _initialize_actors(self, _config):

        # ---- 1. Parked car ----
        # parked_dist = ego front → parked car REAR.
        # Parked car center = ego_half_len + parked_dist + parked_car_half_len.
        # We don't have the actual bbox yet, so use 2.5 m as the placement estimate.
        ego_half_len = self.ego_vehicles[0].bounding_box.extent.x
        _place_half_len_est = 2.5
        park_location, _ = get_location_in_distance_from_wp(
            self._reference_waypoint, self._parked_dist + ego_half_len + _place_half_len_est)
        park_wp = self._wmap.get_waypoint(park_location)

        self._parked_transform = self._get_blocker_transform(park_wp)
        self.parking_slots.append(self._parked_transform.location)

        parked_vehicle = None
        for _ in range(3):
            parked_vehicle = CarlaDataProvider.request_new_actor(
                'vehicle.*', self._parked_transform,
                rolename='scenario', attribute_filter=self._bp_attributes)
            if parked_vehicle is not None:
                break

        if parked_vehicle is None:
            raise ValueError("BtParkedWithBlindSpotPed: failed to spawn parked vehicle")

        parked_vehicle.apply_control(carla.VehicleControl(hand_brake=True))
        parked_vehicle.set_simulate_physics(True)
        self._parked_car_half_len = parked_vehicle.bounding_box.extent.x
        self.other_actors.append(parked_vehicle)  # index 0

        # ---- 2. Pedestrian — positioned using actual bounding box of parked car ----
        walker_dist = parked_vehicle.bounding_box.extent.x + 0.5
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

        # trigger fires when ego front is trigger_dist from parked car rear
        # → radius from collision_x to ego center = trigger_dist + ego_half_len + 2*pchl + 0.5
        trigger_from_center = self._trigger_dist + ego_half_len + 2 * pchl + 0.5

        sequence.add_child(InLaneTriggerDistance(
            self.ego_vehicles[0], collision_location, trigger_from_center,
            self._wmap, name="TriggerAdversaryStart"))

        if self.route_mode:
            sequence.add_child(LeaveCrossingSpace(self._collision_wp))

        # 3x lane_width guarantees the walker clears from shoulder to opposite shoulder
        crossing_dist = self._collision_wp.lane_width * 3.0
        crossing_duration = crossing_dist / self._adversary_speed
        sequence.add_child(KeepVelocity(
            self.other_actors[1],
            self._adversary_speed,
            duration=crossing_duration,
            distance=crossing_dist,
            name="PedWalksAcross"
        ))

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

#!/usr/bin/python3
"""
phase2_ros_controller.py — Phase 2 (CARLA + ROS 2 simple controller) scenario.

Subclass of BtParkedWithBlindSpotPed that makes the pedestrian OPTIONAL, so the same
blind-spot bus geometry can be run in both Phase 2 conditions:

  Phase 2.1 — spawn_ped=false : bus only, no pedestrian. The ROS controller drives the ego
              through the avoidance maneuver; measures closed-loop control repeatability.
  Phase 2.2 — spawn_ped=true  : bus + pedestrian (same as Phase 1B / the old Exp 2). The ROS
              controller avoids the bus AND stops for the pedestrian.

The ego is driven externally by ros_avoid_controller.py (via --waitForEgo); this scenario only
spawns the actors and (when present) triggers the pedestrian walk. It deliberately does NOT touch
the parent BtParkedWithBlindSpotPed — it reuses the parent's inherited geometry helpers
(_get_blocker_transform / _get_walker_transform / _replace_walker / get_location_on_same_road) and
only overrides actor spawning + behavior to honor spawn_ped.

With spawn_ped=true the behavior is identical to the parent (super()._create_behavior()), so
Phase 2.2 is bit-for-bit the current Exp 2 run.
"""
import carla
import py_trees

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (
    ActorDestroy,
    ActorTransformSetter,
)
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import DriveDistance
from srunner.tools.background_manager import LeaveSpaceInFront

from srunner.scenarios.bt_parked_with_blindspot_ped import (
    BtParkedWithBlindSpotPed,
    get_location_on_same_road,
    get_value_parameter,
)


def _as_bool(value):
    """Parse a ScenarioRunner string parameter into a bool ('true'/'1'/'yes' → True)."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ('true', '1', 'yes', 'on')


class Phase2RosController(BtParkedWithBlindSpotPed):

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False,
                 criteria_enable=True, timeout=150):
        # set BEFORE super().__init__ — BasicScenario.__init__ calls _initialize_actors,
        # which reads self._spawn_ped.
        self._spawn_ped = get_value_parameter(config, 'spawn_ped', _as_bool, True)
        super(Phase2RosController, self).__init__(
            world, ego_vehicles, config, randomize, debug_mode, criteria_enable, timeout)

    # ------------------------------------------------------------------
    def _initialize_actors(self, _config):

        # ---- 1. Parked car (always — the ego avoids it in both 2.1 and 2.2) ----
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
            raise ValueError("Phase2RosController: failed to spawn parked vehicle")

        parked_vehicle.apply_control(carla.VehicleControl(hand_brake=True))
        parked_vehicle.set_simulate_physics(True)
        self._parked_car_half_len = parked_vehicle.bounding_box.extent.x
        self.other_actors.append(parked_vehicle)  # index 0

        if not self._spawn_ped:
            return  # Phase 2.1 — bus only, no pedestrian

        # ---- 2. Pedestrian (Phase 2.2 only) — positioned using actual bbox of parked car ----
        walker_dist = parked_vehicle.bounding_box.extent.x + 0.5
        wps = park_wp.next(walker_dist)
        if not wps:
            raise ValueError("Phase2RosController: couldn't find walker waypoint")
        walker_wp = wps[0]
        self._collision_wp = walker_wp

        self._ped_transform = self._get_walker_transform(walker_wp)
        self.parking_slots.append(self._ped_transform.location)

        pedestrian = CarlaDataProvider.request_new_actor('walker.*', self._ped_transform)
        if pedestrian is None:
            raise ValueError("Phase2RosController: failed to spawn pedestrian")

        pedestrian.set_location(self._ped_transform.location + carla.Location(z=-200))
        pedestrian = self._replace_walker(pedestrian)
        self.other_actors.append(pedestrian)  # index 1

    # ------------------------------------------------------------------
    def _create_behavior(self):
        if self._spawn_ped:
            # identical to BtParkedWithBlindSpotPed → Phase 2.2 == old Exp 2
            return super(Phase2RosController, self)._create_behavior()

        # Phase 2.1 — bus avoidance only, no pedestrian event
        sequence = py_trees.composites.Sequence("Phase2RosControllerNoPed", memory=True)

        ego_half_len = self.ego_vehicles[0].bounding_box.extent.x
        pchl = self._parked_car_half_len

        if self.route_mode:
            total_dist = self._parked_dist + ego_half_len + 2 * pchl + 15
            sequence.add_child(LeaveSpaceInFront(total_dist))

        # disable physics on the parked car so the ego can pass without collision/sensor crash
        sequence.add_child(ActorTransformSetter(
            self.other_actors[0], self._parked_transform, physics=False,
            name="DisableParkedCarPhysics"))
        sequence.add_child(DriveDistance(
            self.ego_vehicles[0], self._parked_dist + ego_half_len + 2 * pchl + 10,
            name="EndCondition"))
        sequence.add_child(ActorDestroy(self.other_actors[0], name="DestroyParkedCar"))

        return sequence

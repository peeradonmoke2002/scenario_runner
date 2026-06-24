#!/usr/bin/python3
"""
Scenario 2 — Roadside standing pedestrian (avoid-vs-stop arbitration test).

No parked car. A pedestrian waits on the right sidewalk. When the ego closes to
`trigger_dist` (ego front bbox -> ped bbox), the pedestrian walks off the
sidewalk onto the RIGHT side of the road (right of lane center, NOT the center)
and STOPS there, standing. The ego must arbitrate: avoid (shift left into the
clear oncoming lane) vs stop. After the ego passes, the pedestrian finishes
crossing.

Levers (set in the .xml <other_parameter>s, fixed per run):
  trigger_dist     ego-front -> ped distance when the ped walks out. THE sweep
                   lever: large (>= ~50 m) => ego has room to avoid; small => ego
                   must stop. (Left-lane avoidance is only feasible when the ped
                   is revealed far enough — see thesis notes.)
  ped_stand_offset lateral metres RIGHT of lane center where the ped stands on
                   the road. Larger => intrudes more => tighter avoid. CALIBRATE
                   to the map lane width so it lands in the right part of the lane.
  ped_start_offset lateral metres RIGHT of lane center where the ped starts (on
                   the sidewalk). Must be > ped_stand_offset (= the walk-out gap).
  ped_dist         longitudinal metres from ego start to the ped's road position
                   (like Scn1 parked_dist=100 — lets the ego reach 30 km/h first).
  adversary_speed  walk speed (m/s).
  cross_after_pass 'true' => ped finishes crossing once the ego has driven past.

Assumes the ego route is on the right lane of a road with a (clear) oncoming
lane to the LEFT, so left avoidance is geometrically possible.
"""

import time

import carla
import py_trees

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (
    ActorDestroy,
    ActorTransformSetter,
    KeepVelocity,
    MovePedestrianWithEgo,
    WaitForever,
)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenarios.basic_scenario import BasicScenario
# reuse the trigger + param/road helpers from the blind-spot scenario (DRY)
from srunner.scenarios.bt_parked_with_blindspot_ped import (
    InLaneTriggerDistance,
    get_value_parameter,
    get_location_on_same_road,
)
from srunner.tools.background_manager import LeaveSpaceInFront, LeaveCrossingSpace


class EgoPassedTargetDistance(py_trees.behaviour.Behaviour):
    """SUCCESS once the ego has gone `pass_dist` BEYOND `target_location`.

    `pass_dist` is the lane-projected gap from the ego CENTER to the target center at the
    firing moment; the caller bakes ego_half_len + ped_half into it so the realised gap is
    measured **ego REAR box -> ped RIGHT box edge** (= the distance the ego has passed the
    ped). Uses the ego's lane-projected waypoint (robust to the lateral avoidance shift) and
    tracks the closest approach, so it fires only after the ego has actually passed and
    receded — not while approaching. If the ego stops short and never passes, it never fires
    and the run ends on timeout.
    """

    def __init__(self, ego, target_location, pass_dist, wmap, name="EgoPassedTargetDistance"):
        super().__init__(name)
        self._ego = ego
        self._target_location = target_location
        self._pass_dist = pass_dist
        self._wmap = wmap
        self._min_dist = float('inf')
        self._approached = False

    def update(self):
        ego_wp = self._wmap.get_waypoint(self._ego.get_location())
        d = ego_wp.transform.location.distance(self._target_location)
        if d < self._min_dist:
            self._min_dist = d
        if d < 3.0:                      # ego is alongside the target => it has passed
            self._approached = True
        if self._approached and d > self._min_dist + self._pass_dist:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING


class WaitSeconds(py_trees.behaviour.Behaviour):
    """SUCCESS after `seconds` from the first tick.

    ponytail: wall-clock (time.monotonic), matching InLaneTriggerDistance; fine as a despawn
    cap, not sim-time exact. If CARLA runs far off real-time, scale `seconds` accordingly.
    """

    def __init__(self, seconds, name="WaitSeconds"):
        super().__init__(name)
        self._seconds = seconds
        self._t0 = None

    def update(self):
        now = time.monotonic()
        if self._t0 is None:
            self._t0 = now
        if now - self._t0 >= self._seconds:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING


class BtRoadsideStandingPed(BasicScenario):

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False,
                 criteria_enable=True, timeout=150):

        self._wmap = CarlaDataProvider.get_map()
        self._reference_waypoint = self._wmap.get_waypoint(config.trigger_points[0].location)
        self.timeout = timeout

        self._ped_dist         = get_value_parameter(config, 'ped_dist',         float, 100.0)
        self._trigger_dist     = get_value_parameter(config, 'trigger_dist',     float, 50.0)
        self._adversary_speed  = get_value_parameter(config, 'adversary_speed',  float, 1.2)
        self._ped_stand_offset = get_value_parameter(config, 'ped_stand_offset', float, 1.2)
        self._ped_start_offset = get_value_parameter(config, 'ped_start_offset', float, 3.5)
        self._cross_after_pass = get_value_parameter(config, 'cross_after_pass', str,   'true')
        # gap (ego REAR box -> ped RIGHT box edge) the ego must open up after passing
        # before the ped starts crossing.
        self._ego_pass_dist    = get_value_parameter(config, 'ego_pass_dist',    float, 5.0)
        # seconds the ped stays on the road (from when it reaches its standing spot) before
        # it disappears — caps the wait so it never lingers if the ego stops and never passes.
        self._ped_despawn_time = get_value_parameter(config, 'ped_despawn_time', float, 30.0)

        if self._ped_start_offset <= self._ped_stand_offset:
            raise ValueError(
                "BtRoadsideStandingPed: ped_start_offset must be > ped_stand_offset "
                "(the sidewalk->road walk-out gap)")

        self._stand_wp            = None
        self._ped_stand_location  = None   # road point where the ped ends up standing
        self._ped_start_transform = None   # sidewalk spawn pose

        super(BtRoadsideStandingPed, self).__init__(
            "BtRoadsideStandingPed", ego_vehicles, config, world,
            debug_mode, criteria_enable=criteria_enable)

    # ------------------------------------------------------------------
    def _lateral_location(self, waypoint, right_offset_m):
        """Point `right_offset_m` to the RIGHT of the lane centerline at `waypoint`."""
        rv = waypoint.transform.get_right_vector()
        loc = waypoint.transform.location + carla.Location(
            x=right_offset_m * rv.x, y=right_offset_m * rv.y)
        loc.z += 1.2
        return loc

    # ------------------------------------------------------------------
    def _initialize_actors(self, _config):
        # ped's longitudinal road position (no bus — straight ahead of the ego)
        stand_loc, _ = get_location_on_same_road(self._reference_waypoint, self._ped_dist)
        self._stand_wp = self._wmap.get_waypoint(stand_loc)

        # publish the obstacle x so auto_test.py can compute the goal past the ped
        # (same /tmp file the blind-spot scenario writes for the parked car)
        with open('/tmp/bt_park_actual_x.txt', 'w') as f:
            f.write(f"{self._stand_wp.transform.location.x:.3f}")

        self._ped_stand_location = self._lateral_location(self._stand_wp, self._ped_stand_offset)
        start_loc = self._lateral_location(self._stand_wp, self._ped_start_offset)

        # +270 makes a right-side walker head LEFT (sidewalk -> lane center), the same
        # crossing direction as the blind-spot scenario; KeepVelocity stops it once it has
        # covered the walk-out gap, leaving it standing on the right of the lane.
        road_yaw = self._stand_wp.transform.rotation.yaw
        self._ped_start_transform = carla.Transform(
            start_loc, carla.Rotation(yaw=road_yaw + 270))
        self.parking_slots.append(self._ped_stand_location)

        pedestrian = CarlaDataProvider.request_new_actor('walker.*', self._ped_start_transform)
        if pedestrian is None:
            raise ValueError("BtRoadsideStandingPed: failed to spawn pedestrian")
        pedestrian.set_location(self._ped_start_transform.location + carla.Location(z=-200))
        pedestrian = self._replace_walker(pedestrian)
        self.other_actors.append(pedestrian)  # index 0

    # ------------------------------------------------------------------
    def _create_behavior(self):
        sequence = py_trees.composites.Sequence("BtRoadsideStandingPed", memory=True)
        ped = self.other_actors[0]
        ego = self.ego_vehicles[0]
        ego_half_len = ego.bounding_box.extent.x

        if self.route_mode:
            sequence.add_child(LeaveSpaceInFront(self._ped_dist + ego_half_len + 5))

        sequence.add_child(ActorTransformSetter(
            ped, self._ped_start_transform, True, name="PlacePedestrian"))

        # trigger fires when ego front is trigger_dist from the ped. The InLaneTriggerDistance
        # measures ego-lane-waypoint -> stand_wp center; ego_half_len converts ego center->front
        # and the ped extent converts ped center->near edge, so trigger_dist is box-front-ego ->
        # box-of-ped. Confirm the realised gap in /tmp/bt_trigger_debug/trigger_*.csv.
        ped_box_half = ped.bounding_box.extent.x
        trigger_from_center = self._trigger_dist + ego_half_len + ped_box_half
        sequence.add_child(InLaneTriggerDistance(
            ego, self._stand_wp.transform.location, trigger_from_center,
            self._wmap, name="TriggerPedWalkOut"))

        if self.route_mode:
            sequence.add_child(LeaveCrossingSpace(self._stand_wp))

        # ped walks sidewalk -> right-of-lane standing spot, then KeepVelocity ends -> stands.
        # It must hold still > moving_time_threshold (~1 s) for AW avoidance to target it.
        walk_dist = self._ped_start_offset - self._ped_stand_offset
        sequence.add_child(KeepVelocity(
            ped, self._adversary_speed,
            duration=walk_dist / self._adversary_speed, distance=walk_dist,
            name="PedWalkToRoad"))

        # ego arbitrates avoid/stop while the ped stands. The ped waits until the ego has
        # passed it by ego_pass_dist (ego REAR box -> ped RIGHT box edge), then optionally
        # finishes crossing. pass_threshold converts the lane-center ego->ped gap into that
        # rear-ego -> right-ped-edge measure (add ego_half_len + ped along-road half).
        pass_threshold = self._ego_pass_dist + ego_half_len + ped_box_half
        stand_then_cross = py_trees.composites.Sequence("PedStandThenCross", memory=True)
        stand_then_cross.add_child(EgoPassedTargetDistance(
            ego, self._stand_wp.transform.location, pass_threshold, self._wmap,
            name="EgoPassesPed"))
        if self._cross_after_pass == 'true':
            remaining = self._stand_wp.lane_width * 2.0
            stand_then_cross.add_child(KeepVelocity(
                ped, self._adversary_speed,
                duration=remaining / self._adversary_speed, distance=remaining,
                name="PedFinishCross"))

        # the ped disappears when (ego passes + finishes cross) OR after ped_despawn_time s,
        # whichever comes first. Keep the scenario tree alive after despawn so auto_test.py
        # ends the run from the ego reaching the goal, not from this pedestrian timeout.
        despawn = py_trees.composites.Parallel(
            "PedStandUntilPassOrTimeout",
            policy=py_trees.common.ParallelPolicy.SuccessOnOne())
        despawn.add_child(stand_then_cross)
        despawn.add_child(WaitSeconds(self._ped_despawn_time, name="PedDespawnTimeout"))
        sequence.add_child(despawn)

        sequence.add_child(ActorDestroy(ped, name="DestroyPedestrian"))
        sequence.add_child(WaitForever(name="WaitForGoalReachedExternally"))
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
    # Pedestrian large-map dormancy workaround (same pattern as the blind-spot scenario)
    def _replace_walker(self, walker):
        type_id = walker.type_id
        CarlaDataProvider.remove_actor_by_id(walker.id)
        spawn_transform = self.ego_vehicles[0].get_transform()
        spawn_transform.location.z -= 50
        walker = CarlaDataProvider.request_new_actor(type_id, spawn_transform)
        if not walker:
            raise ValueError("BtRoadsideStandingPed: couldn't spawn walker substitute")
        walker.set_simulate_physics(False)
        walker.set_location(spawn_transform.location + carla.Location(z=-50))
        return walker

    def _setup_scenario_trigger(self, config):
        trigger_tree = super()._setup_scenario_trigger(config)
        if not self.route_mode:
            return trigger_tree
        # keep pedestrian active as ego approaches (large-map fix)
        parallel = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SuccessOnOne(), name="ScenarioTrigger")
        parallel.add_child(MovePedestrianWithEgo(self.ego_vehicles[0], self.other_actors[0], 100))
        parallel.add_child(trigger_tree)
        return parallel

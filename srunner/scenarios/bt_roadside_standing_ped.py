#!/usr/bin/python3
"""
Roadside standing pedestrian scenario.

The ego vehicle drives in the right lane while a visible pedestrian waits on
the right sidewalk. When the longitudinal gap reaches ``trigger_dist``, the
pedestrian walks into the right side of the lane and stops. This leaves the ego
vehicle to either avoid the pedestrian through the clear lane on the left or
stop upstream of the pedestrian.

The pedestrian remains at the standing position until one of two events occurs:

* the ego vehicle passes and opens the configured ``ego_pass_dist`` gap, after
  which the pedestrian finishes crossing; or
* ``ped_wait_timeout`` expires, after which the pedestrian is removed so the ego
  vehicle can continue to the route goal.

The scenario controls actor motion and the pass/timeout release only. Experiment
measurements are recorded from the ego-side Autoware topics by the external test
pipeline.

Configurable parameters:

* ``ped_dist``: longitudinal distance from the ego start to the pedestrian.
* ``trigger_dist``: longitudinal ego-front to pedestrian gap that starts walking.
* ``adversary_speed``: pedestrian walking speed in metres per second.
* ``ped_start_offset``: spawn position on the sidewalk, to the right of the lane
  centerline.
* ``ped_walk_distance``: distance from the spawn position to the standing position.
* ``ped_stand_offset``: standing position to the right of the lane centerline
  (0 = lane centre, half the lane width = lane edge).

  The three placement values satisfy start = stand + walk. Configure at most two of
  them and the third is derived; for example a fixed ``ped_start_offset`` with a swept
  ``ped_walk_distance``.
* ``ped_wait_timeout``: maximum time at the standing position.
* ``ego_pass_dist``: clearance behind the ego that releases the pedestrian.
* ``cross_after_pass``: whether the pedestrian finishes crossing after a pass.
* ``ped_cross``: crossing mode. The pedestrian walks from ``ped_start_offset`` across
  both lanes without stopping, then is removed. Only ``ped_start_offset`` may be set;
  the stand, wait and pass parameters are unused.
* ``ped_response`` (crossing mode only): ``none`` (default, constant speed), ``yield``,
  ``hurry`` or ``freeze``. At ``decision_offset`` right of the lane centerline (default
  the ego-lane edge) the pedestrian checks the ego's distance d. If the ego is moving
  and closer than ``gap_accept_m``, ``yield`` and ``freeze`` stop there and ``hurry``
  speeds up to ``hurry_speed``; otherwise it keeps walking. ``freeze`` is ``yield`` with
  the decision inside the lane (for example 1.4 m), so the stopped pedestrian blocks part
  of the ego path. ``distance`` picks the response from d instead: below ``freeze_dist``
  it freezes, below ``gap_accept_m`` it hurries, otherwise it keeps walking.
  A stopped pedestrian walks on ``startup_time`` after the ego stops before the crossing
  line or passes it. With ``freeze_hold`` > 0 it instead stays frozen ``freeze_hold`` s
  after the ego first stops, unless the ego passes first. ``yield_timeout`` is a deadlock
  guard (no hold only): it runs only while the ego stands still without either cue, and
  never releases the pedestrian beside it.

The road segment must provide a clear lane on the left for avoidance.
"""

import os

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
from srunner.scenariomanager.timer import GameTime
from srunner.scenarios.basic_scenario import BasicScenario
# reuse the trigger + param/road helpers from the blind-spot scenario (DRY)
from srunner.scenarios.bt_parked_with_blindspot_ped import (
    InLaneTriggerDistance,
    get_value_parameter,
    get_location_on_same_road,
)
from srunner.tools.background_manager import LeaveSpaceInFront, LeaveCrossingSpace
# Responding-pedestrian behaviour, shared with Scenario 1 (moved to its own module).
from srunner.scenarios.bt_ped_response import (
    PED_RESPONSES,
    CrossingPedResponse,
    get_longitudinal_bbox_extents,
)


ROADSIDE_OUTCOME_FILE = '/tmp/bt_roadside_outcome.txt'


class EgoPassedTargetDistance(py_trees.behaviour.Behaviour):
    """Return SUCCESS after the ego passes the target by ``pass_dist``.

    ``pass_dist`` is measured along the road from the ego center to the target
    origin. The caller includes the ego and pedestrian bounding-box extents so
    the configured gap represents the space behind the ego. Lateral displacement
    is ignored so an avoidance shift does not prevent pass detection.
    """

    def __init__(self, ego, target_location, pass_dist, road_forward,
                 name="EgoPassedTargetDistance"):
        super().__init__(name)
        self._ego = ego
        self._target_location = target_location
        self._pass_dist = pass_dist
        forward_norm = (
            road_forward.x ** 2 + road_forward.y ** 2 + road_forward.z ** 2
        ) ** 0.5
        if forward_norm <= 1e-6:
            raise ValueError("EgoPassedTargetDistance: road_forward must be non-zero")
        self._road_forward = carla.Vector3D(
            road_forward.x / forward_norm,
            road_forward.y / forward_norm,
            road_forward.z / forward_norm,
        )

    def update(self):
        ego_location = self._ego.get_location()
        dx = ego_location.x - self._target_location.x
        dy = ego_location.y - self._target_location.y
        dz = ego_location.z - self._target_location.z
        longitudinal = (
            dx * self._road_forward.x
            + dy * self._road_forward.y
            + dz * self._road_forward.z
        )
        if longitudinal >= self._pass_dist:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING


class WaitSimSeconds(py_trees.behaviour.Behaviour):
    """Return SUCCESS after a duration measured in ScenarioRunner simulation time."""

    def __init__(self, seconds, name="WaitSimSeconds"):
        super().__init__(name)
        self._seconds = seconds
        self._start_time = None

    def initialise(self):
        self._start_time = GameTime.get_time()

    def update(self):
        if GameTime.get_time() - self._start_time >= self._seconds:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING


class RecordStandOutcome(py_trees.behaviour.Behaviour):
    """Record the pass or timeout event used to release the pedestrian."""

    def __init__(self, outcome, state, name="RecordStandOutcome"):
        super().__init__(name)
        self._outcome = outcome
        self._state = state

    def update(self):
        self._state['outcome'] = self._outcome
        with open(ROADSIDE_OUTCOME_FILE, 'w') as outcome_file:
            outcome_file.write(self._outcome)
        return py_trees.common.Status.SUCCESS


class StandWaitTimedOut(py_trees.behaviour.Behaviour):
    """Return SUCCESS when the pedestrian was released by the observation timeout."""

    def __init__(self, state, name="StandWaitTimedOut"):
        super().__init__(name)
        self._state = state

    def update(self):
        if self._state.get('outcome') == 'ped_wait_timeout':
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE


def get_bool_parameter(config, name, default):
    """Read a strict but user-friendly boolean from ScenarioRunner XML parameters."""
    if name not in config.other_parameters:
        return default
    raw_value = config.other_parameters[name]['value']
    normalized = str(raw_value).strip().lower()
    if normalized in ('true', '1', 'yes', 'on'):
        return True
    if normalized in ('false', '0', 'no', 'off'):
        return False
    raise ValueError(
        f"BtRoadsideStandingPed: '{name}' must be true/false, got '{raw_value}'")


class BtRoadsideStandingPed(BasicScenario):

    """Control a pedestrian that walks from the roadside and stops in the lane."""

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False,
                 criteria_enable=True, timeout=600):

        self._wmap = CarlaDataProvider.get_map()
        self._reference_waypoint = self._wmap.get_waypoint(config.trigger_points[0].location)
        self.timeout = timeout

        self._ped_dist         = get_value_parameter(config, 'ped_dist',         float, 100.0)
        self._trigger_dist     = get_value_parameter(config, 'trigger_dist',     float, 50.0)
        self._adversary_speed  = get_value_parameter(config, 'adversary_speed',  float, 1.2)
        self._ped_cross = get_bool_parameter(config, 'ped_cross', False)
        # Placement: start = stand + walk. Set at most two of the three; the third is derived.
        has_start_offset = 'ped_start_offset' in config.other_parameters
        has_walk_distance = 'ped_walk_distance' in config.other_parameters
        has_stand_offset = 'ped_stand_offset' in config.other_parameters
        if self._ped_cross:
            if has_walk_distance or has_stand_offset:
                raise ValueError(
                    "BtRoadsideStandingPed: ped_cross sets only ped_start_offset; "
                    "the pedestrian never stands in the lane")
            self._ped_start_offset = get_value_parameter(
                config, 'ped_start_offset', float, 4.0)
            # Placeholders for the shared checks; the crossing length needs the lane width.
            self._ped_walk_distance = self._ped_start_offset
            self._ped_stand_offset = 0.0
        elif has_start_offset and has_walk_distance and has_stand_offset:
            raise ValueError(
                "BtRoadsideStandingPed: specify at most two of ped_start_offset, "
                "ped_walk_distance and ped_stand_offset; the third is derived")
        if has_start_offset and has_walk_distance:
            # Fixed spawn point, walk length chosen: the stand position follows.
            self._ped_start_offset = get_value_parameter(
                config, 'ped_start_offset', float, 3.5)
            self._ped_walk_distance = get_value_parameter(
                config, 'ped_walk_distance', float, 2.3)
            self._ped_stand_offset = self._ped_start_offset - self._ped_walk_distance
        elif has_walk_distance:
            self._ped_stand_offset = get_value_parameter(
                config, 'ped_stand_offset', float, 1.2)
            self._ped_walk_distance = get_value_parameter(
                config, 'ped_walk_distance', float, 2.3)
            self._ped_start_offset = self._ped_stand_offset + self._ped_walk_distance
        else:
            self._ped_stand_offset = get_value_parameter(
                config, 'ped_stand_offset', float, 1.2)
            self._ped_start_offset = get_value_parameter(
                config, 'ped_start_offset', float, 3.5)
            self._ped_walk_distance = self._ped_start_offset - self._ped_stand_offset
        self._ped_response = str(config.other_parameters.get(
            'ped_response', {'value': 'none'})['value']).strip().lower()
        if self._ped_response not in PED_RESPONSES:
            raise ValueError(
                f"BtRoadsideStandingPed: ped_response must be one of {PED_RESPONSES}, "
                f"got '{self._ped_response}'")
        if self._ped_response != 'none' and not self._ped_cross:
            raise ValueError("BtRoadsideStandingPed: ped_response needs ped_cross")
        # 33 m = a 4 s gap at the 8.33 m/s cruise speed.
        self._gap_accept_m  = get_value_parameter(config, 'gap_accept_m',  float, 33.0)
        # 25 m = a 3 s gap at the 8.33 m/s cruise speed; used only by the distance response.
        self._freeze_dist   = get_value_parameter(config, 'freeze_dist',   float, 25.0)
        if not 0.0 < self._freeze_dist <= self._gap_accept_m:
            raise ValueError("BtRoadsideStandingPed: need 0 < freeze_dist <= gap_accept_m")
        # > 0: a stopped pedestrian stays frozen this long after the ego stops (0 = resume
        # startup_time after the ego stops).
        self._freeze_hold   = get_value_parameter(config, 'freeze_hold',   float, 0.0)
        if self._freeze_hold < 0.0:
            raise ValueError("BtRoadsideStandingPed: freeze_hold must be >= 0")
        self._hurry_speed   = get_value_parameter(config, 'hurry_speed',   float, 2.0)
        self._startup_time  = get_value_parameter(config, 'startup_time',  float, 2.0)
        self._yield_timeout = get_value_parameter(config, 'yield_timeout', float, 5.0)
        # None = the ego-lane edge, resolved once the lane width is known.
        self._decision_offset = (get_value_parameter(config, 'decision_offset', float, 0.0)
                                 if 'decision_offset' in config.other_parameters else None)
        if min(self._gap_accept_m, self._hurry_speed, self._yield_timeout) <= 0.0 \
                or self._startup_time < 0.0:
            raise ValueError(
                "BtRoadsideStandingPed: gap_accept_m, hurry_speed and yield_timeout must be > 0 "
                "and startup_time >= 0")
        self._cross_after_pass = get_bool_parameter(config, 'cross_after_pass', True)
        # Gap behind the ego before the pedestrian resumes crossing.
        self._ego_pass_dist    = get_value_parameter(config, 'ego_pass_dist',    float, 5.0)
        self._ped_wait_timeout = get_value_parameter(config, 'ped_wait_timeout', float, 15.0)
        if self._ped_dist <= 0.0:
            raise ValueError("BtRoadsideStandingPed: ped_dist must be > 0")
        if self._trigger_dist < 0.0:
            raise ValueError("BtRoadsideStandingPed: trigger_dist must be >= 0")
        if self._adversary_speed <= 0.0:
            raise ValueError("BtRoadsideStandingPed: adversary_speed must be > 0")
        if self._ped_stand_offset < 0.0:
            raise ValueError("BtRoadsideStandingPed: ped_stand_offset must be >= 0 (0 = lane centre)")
        if self._ped_walk_distance <= 0.0:
            raise ValueError("BtRoadsideStandingPed: ped_walk_distance must be > 0")
        if self._ped_start_offset <= self._ped_stand_offset:
            raise ValueError(
                "BtRoadsideStandingPed: ped_start_offset must be > ped_stand_offset "
                "(the sidewalk->road walk-out gap)")
        if self._ego_pass_dist < 0.0:
            raise ValueError("BtRoadsideStandingPed: ego_pass_dist must be >= 0")
        if self._ped_wait_timeout <= 0.0:
            raise ValueError("BtRoadsideStandingPed: ped_wait_timeout must be > 0")
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
        # Place the pedestrian longitudinally ahead of the ego on the same road.
        stand_loc, traveled = get_location_on_same_road(
            self._reference_waypoint, self._ped_dist)
        if traveled + 0.5 < self._ped_dist:
            raise ValueError(
                "BtRoadsideStandingPed: ped_dist leaves the reference road before "
                f"the requested position ({traveled:.1f} m reached of {self._ped_dist:.1f} m)")
        self._stand_wp = self._wmap.get_waypoint(stand_loc)
        half_lane_width = 0.5 * self._stand_wp.lane_width
        if self._ped_stand_offset > half_lane_width + 1e-6:
            raise ValueError(
                "BtRoadsideStandingPed: ped_stand_offset must put the pedestrian center "
                f"inside the right half of the lane (<= {half_lane_width:.2f} m)")
        if self._ped_start_offset <= half_lane_width:
            raise ValueError(
                "BtRoadsideStandingPed: ped_start_offset must begin outside the driving lane "
                f"(> {half_lane_width:.2f} m)")

        # Reuse the shared position file so auto_test.py can place the route goal
        # beyond the pedestrian.
        with open('/tmp/bt_park_actual_x.txt', 'w') as f:
            f.write(f"{self._stand_wp.transform.location.x:.3f}")
        try:
            os.unlink(ROADSIDE_OUTCOME_FILE)
        except FileNotFoundError:
            pass

        self._ped_stand_location = self._lateral_location(self._stand_wp, self._ped_stand_offset)
        start_loc = self._lateral_location(self._stand_wp, self._ped_start_offset)

        # Turn the pedestrian from the right sidewalk toward the lane center.
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

        # Convert the requested edge-to-edge trigger gap to the center-based
        # threshold used by InLaneTriggerDistance.
        road_forward = self._stand_wp.transform.get_forward_vector()
        ped_bbox_upstream, ped_bbox_downstream = get_longitudinal_bbox_extents(
            ped, self._ped_start_transform, road_forward)
        trigger_from_center = self._trigger_dist + ego_half_len + ped_bbox_upstream
        sequence.add_child(InLaneTriggerDistance(
            ego, self._stand_wp.transform.location, trigger_from_center,
            self._wmap, name="TriggerPedWalkOut"))

        if self.route_mode:
            sequence.add_child(LeaveCrossingSpace(self._stand_wp))

        if self._ped_cross:
            # Walk across both lanes, to the far edge of the left lane, without stopping.
            # The marker is written at walk onset: the ego may reach its goal first.
            cross_dist = self._ped_start_offset + 1.5 * self._stand_wp.lane_width
            sequence.add_child(RecordStandOutcome(
                'ped_crossing', {'outcome': None}, name="RecordPedCrossing"))
            if self._ped_response != 'none':
                # Decide at decision_offset (default the ego-lane edge). The CARLA walker
                # stops within a few centimetres, so a stopping pedestrian stands there.
                decision_offset = (0.5 * self._stand_wp.lane_width
                                   if self._decision_offset is None else self._decision_offset)
                decision_walk = self._ped_start_offset - decision_offset
                if decision_offset < 0.0 or decision_walk <= 0.0:
                    raise ValueError(
                        "BtRoadsideStandingPed: decision_offset must lie between the lane "
                        f"centre and ped_start_offset (got {decision_offset:.2f} m)")
                sequence.add_child(CrossingPedResponse(
                    ped, ego, self._stand_wp.transform.location, road_forward,
                    self._ped_start_transform.get_forward_vector(),
                    {'response': self._ped_response, 'walk_speed': self._adversary_speed,
                     'hurry_speed': self._hurry_speed, 'gap_accept_m': self._gap_accept_m,
                     'freeze_dist': self._freeze_dist, 'freeze_hold': self._freeze_hold,
                     'startup_time': self._startup_time,
                     'yield_timeout': self._yield_timeout,
                     'decision_walk': decision_walk, 'cross_dist': cross_dist,
                     'ego_half_len': ego_half_len, 'ped_upstream': ped_bbox_upstream,
                     'ped_downstream': ped_bbox_downstream},
                    name="PedCrossesWithResponse"))
                sequence.add_child(ActorDestroy(ped, name="DestroyPedestrian"))
                sequence.add_child(WaitForever(name="WaitForGoalReachedExternally"))
                return sequence
            sequence.add_child(KeepVelocity(
                ped, self._adversary_speed,
                duration=cross_dist / self._adversary_speed,
                distance=cross_dist,
                name="PedWalksAcross"))
            sequence.add_child(ActorDestroy(ped, name="DestroyPedestrian"))
            sequence.add_child(WaitForever(name="WaitForGoalReachedExternally"))
            return sequence

        # Walk from the sidewalk to the configured standing position, then stop.
        sequence.add_child(KeepVelocity(
            ped, self._adversary_speed,
            duration=self._ped_walk_distance / self._adversary_speed,
            distance=self._ped_walk_distance,
            name="PedWalkToRoad"))

        # Keep the pedestrian standing until the ego passes or the observation
        # timeout expires. The external runner still requires the route goal.
        pass_threshold = self._ego_pass_dist + ego_half_len + ped_bbox_downstream
        stand_outcome = {'outcome': None}

        ego_passed = py_trees.composites.Sequence("EgoPassedStandingPed", memory=True)
        ego_passed.add_child(EgoPassedTargetDistance(
            ego, self._stand_wp.transform.location, pass_threshold, road_forward,
            name="EgoPassesPed"))
        ego_passed.add_child(RecordStandOutcome(
            'ego_passed', stand_outcome, name="RecordEgoPassed"))

        wait_timed_out = py_trees.composites.Sequence("StandingPedWaitTimedOut", memory=True)
        wait_timed_out.add_child(WaitSimSeconds(
            self._ped_wait_timeout, name="PedWaitTimeout"))
        wait_timed_out.add_child(RecordStandOutcome(
            'ped_wait_timeout', stand_outcome, name="RecordPedWaitTimeout"))

        wait_for_pass_or_timeout = py_trees.composites.Parallel(
            "WaitForEgoPassOrTimeout",
            policy=py_trees.common.ParallelPolicy.SuccessOnOne())
        wait_for_pass_or_timeout.add_child(ego_passed)
        wait_for_pass_or_timeout.add_child(wait_timed_out)
        sequence.add_child(wait_for_pass_or_timeout)

        if self._cross_after_pass:
            remaining = self._stand_wp.lane_width * 2.0
            finish_or_remove = py_trees.composites.Selector(
                "FinishCrossUnlessWaitTimedOut", memory=True)
            finish_or_remove.add_child(StandWaitTimedOut(stand_outcome))
            finish_or_remove.add_child(KeepVelocity(
                ped, self._adversary_speed,
                duration=remaining / self._adversary_speed, distance=remaining,
                name="PedFinishCross"))
            sequence.add_child(finish_or_remove)

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
    # Keep the pedestrian actor active on the large map before scenario placement.
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
        # Move the hidden pedestrian with the ego until the route trigger activates.
        parallel = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SuccessOnOne(), name="ScenarioTrigger")
        parallel.add_child(MovePedestrianWithEgo(self.ego_vehicles[0], self.other_actors[0], 100))
        parallel.add_child(trigger_tree)
        return parallel

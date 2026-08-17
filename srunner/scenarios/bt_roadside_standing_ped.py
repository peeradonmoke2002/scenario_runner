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
                   the road. Smaller => intrudes more => tighter avoid. CALIBRATE
                   to the map lane width so it lands in the right part of the lane.
  ped_start_offset lateral metres RIGHT of lane center where the ped starts (on
                   the sidewalk). Used when ped_walk_distance is not supplied.
  ped_walk_distance fixed sidewalk-to-stand travel distance. Prefer this during
                   a stand-offset sweep so every condition has the same walk time.
  ped_dist         longitudinal metres from ego start to the ped's road position
                   (like Scn1 parked_dist=100 — lets the ego reach 30 km/h first).
  adversary_speed  walk speed (m/s).
  cross_after_pass 'true' => ped finishes crossing once the ego has driven past.
  The scenario has no ego-pass deadline. The pedestrian stays in place until the
  ego genuinely passes; the external test runner ends the run at the route goal
  (or reports its own experiment timeout if the ego remains blocked).

Assumes the ego route is on the right lane of a road with a (clear) oncoming
lane to the LEFT, so left avoidance is geometrically possible.
"""

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

    `pass_dist` is the signed road-longitudinal gap from the ego CENTER to the target origin;
    the caller adds ego rear and pedestrian downstream bounding-box extents so the realised
    gap is measured **ego rear -> pedestrian downstream edge**. Lateral displacement is
    deliberately ignored, so shifting into the oncoming lane cannot prevent pass detection.
    A negative projection means that the ego is still approaching the target.
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


def get_longitudinal_bbox_extents(actor, actor_transform, road_forward):
    """Return actor-origin -> upstream/downstream OBB extents along the road."""
    forward_norm = (
        road_forward.x ** 2 + road_forward.y ** 2 + road_forward.z ** 2
    ) ** 0.5
    if forward_norm <= 1e-6:
        raise ValueError("get_longitudinal_bbox_extents: road_forward must be non-zero")
    road_forward = carla.Vector3D(
        road_forward.x / forward_norm,
        road_forward.y / forward_norm,
        road_forward.z / forward_norm,
    )
    actor_forward = actor_transform.get_forward_vector()
    actor_right = actor_transform.get_right_vector()
    actor_up = actor_transform.get_up_vector()

    def dot(vector):
        return (
            vector.x * road_forward.x
            + vector.y * road_forward.y
            + vector.z * road_forward.z
        )

    bbox = actor.bounding_box
    center_projection = (
        bbox.location.x * dot(actor_forward)
        + bbox.location.y * dot(actor_right)
        + bbox.location.z * dot(actor_up)
    )
    half_projection = (
        abs(dot(actor_forward)) * bbox.extent.x
        + abs(dot(actor_right)) * bbox.extent.y
        + abs(dot(actor_up)) * bbox.extent.z
    )
    upstream = max(0.0, half_projection - center_projection)
    downstream = max(0.0, half_projection + center_projection)
    return upstream, downstream


class BtRoadsideStandingPed(BasicScenario):

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False,
                 criteria_enable=True, timeout=600):

        self._wmap = CarlaDataProvider.get_map()
        self._reference_waypoint = self._wmap.get_waypoint(config.trigger_points[0].location)
        self.timeout = timeout

        self._ped_dist         = get_value_parameter(config, 'ped_dist',         float, 100.0)
        self._trigger_dist     = get_value_parameter(config, 'trigger_dist',     float, 50.0)
        self._adversary_speed  = get_value_parameter(config, 'adversary_speed',  float, 1.2)
        self._ped_stand_offset = get_value_parameter(config, 'ped_stand_offset', float, 1.2)
        configured_start_offset = get_value_parameter(
            config, 'ped_start_offset', float, 3.5)
        if 'ped_walk_distance' in config.other_parameters:
            self._ped_walk_distance = get_value_parameter(
                config, 'ped_walk_distance', float, 2.3)
            self._ped_start_offset = self._ped_stand_offset + self._ped_walk_distance
        else:
            self._ped_start_offset = configured_start_offset
            self._ped_walk_distance = self._ped_start_offset - self._ped_stand_offset
        self._cross_after_pass = get_bool_parameter(config, 'cross_after_pass', True)
        # gap (ego REAR box -> pedestrian downstream box edge) to open after passing
        # before the ped starts crossing.
        self._ego_pass_dist    = get_value_parameter(config, 'ego_pass_dist',    float, 5.0)
        if self._ped_dist <= 0.0:
            raise ValueError("BtRoadsideStandingPed: ped_dist must be > 0")
        if self._trigger_dist < 0.0:
            raise ValueError("BtRoadsideStandingPed: trigger_dist must be >= 0")
        if self._adversary_speed <= 0.0:
            raise ValueError("BtRoadsideStandingPed: adversary_speed must be > 0")
        if self._ped_stand_offset <= 0.0:
            raise ValueError("BtRoadsideStandingPed: ped_stand_offset must be > 0")
        if self._ped_walk_distance <= 0.0:
            raise ValueError("BtRoadsideStandingPed: ped_walk_distance must be > 0")
        if self._ped_start_offset <= self._ped_stand_offset:
            raise ValueError(
                "BtRoadsideStandingPed: ped_start_offset must be > ped_stand_offset "
                "(the sidewalk->road walk-out gap)")
        if self._ego_pass_dist < 0.0:
            raise ValueError("BtRoadsideStandingPed: ego_pass_dist must be >= 0")
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
        stand_loc, traveled = get_location_on_same_road(
            self._reference_waypoint, self._ped_dist)
        if traveled + 0.5 < self._ped_dist:
            raise ValueError(
                "BtRoadsideStandingPed: ped_dist leaves the reference road before "
                f"the requested position ({traveled:.1f} m reached of {self._ped_dist:.1f} m)")
        self._stand_wp = self._wmap.get_waypoint(stand_loc)
        half_lane_width = 0.5 * self._stand_wp.lane_width
        if self._ped_stand_offset >= half_lane_width:
            raise ValueError(
                "BtRoadsideStandingPed: ped_stand_offset must put the pedestrian center "
                f"inside the right half of the lane (< {half_lane_width:.2f} m)")
        if self._ped_start_offset <= half_lane_width:
            raise ValueError(
                "BtRoadsideStandingPed: ped_start_offset must begin outside the driving lane "
                f"(> {half_lane_width:.2f} m)")

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

        # Trigger fires at the requested road-longitudinal bbox gap. The walker faces across
        # the road, so blindly using bbox.extent.x would use the wrong axis. Project its full
        # oriented bounding box onto the road direction instead.
        road_forward = self._stand_wp.transform.get_forward_vector()
        ped_bbox_upstream, ped_bbox_downstream = get_longitudinal_bbox_extents(
            ped, self._ped_start_transform, road_forward)
        trigger_from_center = self._trigger_dist + ego_half_len + ped_bbox_upstream
        sequence.add_child(InLaneTriggerDistance(
            ego, self._stand_wp.transform.location, trigger_from_center,
            self._wmap, name="TriggerPedWalkOut"))

        if self.route_mode:
            sequence.add_child(LeaveCrossingSpace(self._stand_wp))

        # ped walks sidewalk -> right-of-lane standing spot, then KeepVelocity ends -> stands.
        # It must hold still > moving_time_threshold (~1 s) for AW avoidance to target it.
        sequence.add_child(KeepVelocity(
            ped, self._adversary_speed,
            duration=self._ped_walk_distance / self._adversary_speed,
            distance=self._ped_walk_distance,
            name="PedWalkToRoad"))

        # Ego arbitrates avoid/stop while the pedestrian stands. Wait indefinitely for a
        # genuine signed longitudinal pass. auto_test.py owns the experiment deadline and
        # only reports success at the route goal; this scenario must not end a run early.
        pass_threshold = self._ego_pass_dist + ego_half_len + ped_bbox_downstream
        sequence.add_child(EgoPassedTargetDistance(
            ego, self._stand_wp.transform.location, pass_threshold, road_forward,
            name="EgoPassesPed"))

        if self._cross_after_pass:
            remaining = self._stand_wp.lane_width * 2.0
            sequence.add_child(KeepVelocity(
                ped, self._adversary_speed,
                duration=remaining / self._adversary_speed, distance=remaining,
                name="PedFinishCross"))

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

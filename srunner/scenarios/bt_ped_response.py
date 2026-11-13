#!/usr/bin/python3
"""
Responding-pedestrian behaviour shared by the BT pedestrian scenarios.

``CrossingPedResponse`` walks a pedestrian across the road and, at a decision point, makes it
freeze, hurry or keep walking depending on the ego distance (see the class docstring). It was
moved here unchanged from ``bt_roadside_standing_ped.py`` (Scenario 2, Experiment 2) so that
``bt_parked_with_blindspot_ped.py`` (Scenario 1, Experiment 3) can use the same rules without a
circular import. This module defines no scenario class.
"""

import csv
import math

import carla
import py_trees

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime

# Per-tick log of the responding pedestrian; auto_test.py copies it into the bag folder.
PED_RESPONSE_LOG = '/tmp/bt_ped_response.csv'
PED_RESPONSES = ('none', 'yield', 'hurry', 'freeze', 'distance')
# Responses that stop at the decision point and wait for the ego.
STOP_RESPONSES = ('yield', 'freeze')
# Below this speed the ego counts as stopped.
EGO_STOPPED_SPEED = 0.1


def time_to_arrival(d, v):
    """Constant-speed time for the ego front to cover ``d`` metres to the crossing line.

    Returns 0 when the ego is at or past the line and infinity when it is stopped.
    """
    if d <= 0.0:
        return 0.0
    if v < EGO_STOPPED_SPEED:
        return math.inf
    return d / v


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


class CrossingPedResponse(py_trees.behaviour.Behaviour):
    """Walk the pedestrian across the road with a response at the decision point.

    One behaviour drives the walker every tick, so no gap between segments lets the
    walker slow down (KeepVelocity.terminate sets the walker speed to 0).

    Phases: ``approach`` (walk to the decision point) -> gap check -> ``continue``,
    ``hurry``, ``yield`` or ``freeze``. A stopping pedestrian waits for the ego to stop
    before the crossing line or to pass it, waits ``startup_time``, then walks on.
    ``yield_timeout`` counts only while the ego stands still without a cue, so a slowly
    approaching ego never releases the pedestrian early. SUCCESS once ``cross_dist`` is
    walked.

    Ego distances are measured along ``road_forward`` from the crossing line through
    the pedestrian origin. ``d`` is the gap from the ego front to the near edge of the
    pedestrian box, matching the edge-to-edge ``trigger_dist``.
    """

    def __init__(self, ped, ego, line_location, road_forward, cross_direction, params,
                 name="CrossingPedResponse"):
        super().__init__(name)
        self._ped = ped
        self._ego = ego
        self._line = line_location
        norm = math.sqrt(road_forward.x ** 2 + road_forward.y ** 2)
        self._forward = (road_forward.x / norm, road_forward.y / norm)
        norm = math.sqrt(cross_direction.x ** 2 + cross_direction.y ** 2)
        self._cross_dir = carla.Vector3D(cross_direction.x / norm, cross_direction.y / norm, 0.0)
        self._p = params
        self._start = None
        self._phase = None
        self._speed = 0.0
        self._t_still = None
        self._t_cue = None
        self._t_hold = None
        self._log = None
        self._writer = None

    def initialise(self):
        self._start = self._ped.get_location()
        self._phase = 'approach'
        self._speed = self._p['walk_speed']
        self._t_still = None
        self._t_cue = None
        self._t_hold = None
        self._log = open(PED_RESPONSE_LOG, 'w', newline='')
        self._writer = csv.writer(self._log)
        self._writer.writerow(['t', 'phase', 'walked', 'cmd_speed', 'ego_d', 'ego_v', 'tta', 'event'])

    def _walked(self):
        loc = self._ped.get_location()
        return ((loc.x - self._start.x) * self._cross_dir.x
                + (loc.y - self._start.y) * self._cross_dir.y)

    def _ego_state(self):
        """(d, v, passed): front-to-pedestrian gap, speed, rear past the pedestrian."""
        loc = self._ego.get_location()
        along = ((loc.x - self._line.x) * self._forward[0]
                 + (loc.y - self._line.y) * self._forward[1])
        d = -along - self._p['ego_half_len'] - self._p['ped_upstream']
        passed = along >= self._p['ego_half_len'] + self._p['ped_downstream']
        return d, CarlaDataProvider.get_velocity(self._ego), passed

    def _walk(self, speed):
        self._ped.apply_control(carla.WalkerControl(direction=self._cross_dir, speed=speed))

    def update(self):
        now = GameTime.get_time()
        walked = self._walked()
        d, v, passed = self._ego_state()
        tta = time_to_arrival(d, v)
        event = ''

        if self._phase == 'approach' and walked >= self._p['decision_walk']:
            # Respond only to a moving ego inside the accepted gap; TTA is logged, not used.
            if passed or v < EGO_STOPPED_SPEED or d >= self._p['gap_accept_m']:
                self._phase = 'continue'
            elif self._p['response'] == 'distance':
                self._phase = 'freeze' if d < self._p['freeze_dist'] else 'hurry'
            else:
                self._phase = self._p['response']
            if self._phase == 'hurry':
                self._speed = self._p['hurry_speed']
            elif self._phase in STOP_RESPONSES:
                self._speed = 0.0
            event = f'decision_{self._phase}'
        elif self._phase in STOP_RESPONSES:
            hold = self._p['freeze_hold']
            if self._t_cue is None:
                if passed:
                    self._t_cue, event = now, 'cue_ego_passed'
                elif hold > 0.0:
                    # Hold: stay frozen for `hold` s after the ego first stops, so a stopped
                    # ego has time to show whether it waits or avoids.
                    if v < EGO_STOPPED_SPEED and self._t_hold is None:
                        self._t_hold, event = now, 'hold_start'
                elif v < EGO_STOPPED_SPEED and d > 0.0:
                    self._t_cue, event = now, 'cue_ego_stopped'
            # Deadlock timer: runs only while the ego stands still without a cue or hold.
            if self._t_cue is None and hold <= 0.0 and v < EGO_STOPPED_SPEED:
                self._t_still = self._t_still if self._t_still is not None else now
            else:
                self._t_still = None
            if self._t_cue is not None and now - self._t_cue >= self._p['startup_time']:
                self._phase, self._speed = 'resume', self._p['walk_speed']
                event = 'resume_after_cue'
            elif (self._t_cue is None and self._t_hold is not None
                  and now - self._t_hold >= hold and not (d < 0.0 and not passed)):
                self._phase, self._speed = 'resume', self._p['walk_speed']
                event = 'resume_after_hold'
            elif (self._t_still is not None
                  and now - self._t_still >= self._p['yield_timeout']
                  and not (d < 0.0 and not passed)):
                # Never cuts a start-up short and never walks the pedestrian into the side
                # of an ego that is still beside it.
                self._phase, self._speed = 'resume', self._p['walk_speed']
                event = 'resume_after_timeout'

        self._walk(self._speed)
        self._writer.writerow([f'{now:.3f}', self._phase, f'{walked:.3f}', f'{self._speed:.2f}',
                               f'{d:.3f}', f'{v:.3f}',
                               'inf' if math.isinf(tta) else f'{tta:.3f}', event])
        if walked >= self._p['cross_dist']:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        if self._log is not None:
            self._log.close()
            self._log = None
        super().terminate(new_status)

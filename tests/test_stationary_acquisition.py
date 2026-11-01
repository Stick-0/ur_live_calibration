"""Stationary consensus, checked handoff holds, and recorded noisy PnP replay."""

from contextlib import redirect_stderr
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from tests.test_active_calibration import packet
import tests.test_live_alignment_runtime as lifecycle
from urcal.control.acquisition import StationaryCalibrationAcquisition
from urcal.control.geometry import flange_from_tcp, inverse
from urcal.control.handoff import CalibrationHandoff
from urcal.control.policy import CalibrationFrame, CalibrationTracker, ControlFault
from urcal.control.runtime import parse_args
from urcal.control.startup import check_alignment_request, check_handoff_snapshot
from urcal.cooperative.controller import CooperativeController


def options():
    return parse_args(['--mode', 'follow', '--ur5e_ip', '192.0.2.5', '--ur10e_ip', '192.0.2.10',
                       '--initial_align'])


class StationaryAcquisitionTests(unittest.TestCase):
    def acquisition(self):
        return StationaryCalibrationAcquisition(CalibrationTracker(
            max_jump_m=.05, max_jump_rad=np.deg2rad(5), check_reprojection=False))

    def test_good_frames_survive_isolated_bad_images(self):
        acquisition = self.acquisition()
        for i in range(7):
            at = 10. + i * .08
            candidate = acquisition.offer(packet(i, at, reason='PnP failed' if i in (2, 4) else ''), at)
        candidate.check_fresh(at)
        self.assertEqual(candidate.stable_count, 5)
        self.assertEqual(candidate.last_capture, at)

    def test_outlier_never_refreshes_selected_image_time_or_changes_pose(self):
        acquisition = self.acquisition()
        for i in range(5):
            previous = acquisition.offer(packet(i, 10. + .08*i), 10. + .08*i)
        result = acquisition.offer(packet(5, 10.4, .5), 10.4)
        self.assertEqual(result.last_capture, previous.last_capture)
        np.testing.assert_array_equal(result.poses['ur5e'], previous.poses['ur5e'])
        with self.assertRaisesRegex(ControlFault, 'stale'):
            result.check_fresh(10.9)

    def test_ambiguous_clusters_and_chained_drift_do_not_produce_readiness(self):
        for values in ([0., .2]*5, [i*.03 for i in range(10)]):
            acquisition = self.acquisition()
            for i, value in enumerate(values):
                at = 10. + i*.08
                self.assertIsNone(acquisition.offer(packet(i, at, value), at))

    def test_duplicates_do_not_add_evidence_or_revoke_fresh_consensus(self):
        acquisition = self.acquisition()
        for i in range(5):
            result = acquisition.offer(packet(i, 10.+i*.08), 10.+i*.08)
        repeated = acquisition.offer(packet(4, 10.32), 10.4)
        self.assertEqual(repeated.stable_count, 5)
        self.assertEqual(repeated.last_capture, result.last_capture)
        self.assertIsNone(acquisition.offer(packet(3, 10.5), 10.5))
        # Replaying an invalid sample with a repaired payload cannot count.
        acquisition.offer(packet(5, 10.6, reason='PnP failed'), 10.6)
        repeated = acquisition.offer(packet(5, 10.6), 10.61)
        self.assertEqual(repeated.last_capture, result.last_capture)
        self.assertIsNone(acquisition.offer(packet(5, 10.6), 11.2))

    def test_future_stale_and_invalid_quality_do_not_seed_consensus(self):
        for problem in ('future', 'stale', 'quality'):
            acquisition = self.acquisition()
            for i in range(6):
                at = 10. + i*.08
                capture = at + .1 if problem == 'future' else at - .6 if problem == 'stale' else at
                frame = packet(i, capture)
                if problem == 'quality':
                    frame.valid_counts['ur5e'] = 3
                self.assertIsNone(acquisition.offer(frame, at))

    def test_new_stationary_base_is_acquired_after_old_window_expires(self):
        acquisition = self.acquisition()
        for i in range(5):
            acquisition.offer(packet(i, 10.+i*.08), 10.+i*.08)
        for i in range(5):
            at = 12. + i*.08
            result = acquisition.offer(packet(10+i, at, .2), at)
            self.assertEqual(result is not None, i == 4)
        self.assertAlmostEqual(result.poses['ur5e'][0, 3], packet(99, 12.32, .2).poses['ur5e'][0, 3])

    def test_consensus_is_only_used_in_unarmed_initial_alignment(self):
        args = options()
        args.base_motion_mode = "velocity"
        engine = CooperativeController(args)
        for i, value in enumerate((0., .03, 0., .03, 0., .03)):
            at = 10. + .08*i
            engine.ingest(packet(i, at, value), at, allow_reacquire=True)
        snapshot = engine.state(at, active=False)
        for twist in snapshot.base_twists.values():
            np.testing.assert_array_equal(twist, np.zeros(6))
        # In live tracking the original motion rejection remains in force.
        self.assertFalse(engine.ingest(packet(6, at+.01, .04), at+.01, allow_reacquire=False))
        self.assertIn('base linear speed', engine.last_rejection)

    def test_first_live_update_uses_actual_raw_observation_for_velocity(self):
        args = options()
        args.base_motion_mode = "velocity"
        engine = CooperativeController(args)
        for i, value in enumerate((0., .03, 0., .03, .03)):
            at = 10. + .08*i
            engine.ingest(packet(i, at, value), at, allow_reacquire=True)
        # The image is unchanged, though the consensus mean differs by 12mm.
        # At 10ms that discrepancy must not turn into a fictitious 1.2m/s.
        self.assertTrue(engine.ingest(packet(5, at+.01, .03), at+.01, allow_reacquire=False))
        snapshot = engine.state(at+.01, True)
        for twist in snapshot.base_twists.values():
            np.testing.assert_array_equal(twist, np.zeros(6))
        with self.assertRaisesRegex(ValueError, 'empty estimator'):
            engine.estimator.initialize_stationary_pair({}, {}, at, at, 9)

    def test_consensus_checks_relative_bases_as_well_as_each_robot(self):
        acquisition = self.acquisition()
        for i in range(8):
            at = 10.+i*.08
            frame = packet(i, at, .03 if i%2 else 0.)
            frame.poses['ur10e'][0, 3] -= .03 if i%2 else 0.
            self.assertIsNone(acquisition.offer(frame, at))

    def test_timeout_hold_cannot_start_from_old_frames_and_stop_option_remains(self):
        for hold in (True, False):
            handoff = CalibrationHandoff(CalibrationTracker(check_reprojection=False), 10.,
                                         timeout=1., consensus=True, wait_at_rest=hold)
            for i in range(5):
                self.assertFalse(handoff.ingest(packet(i, 9.+i*.08), 10.1))
            if not hold:
                with self.assertRaisesRegex(ControlFault, 'wait exceeded'):
                    handoff.ready(12.)
                continue
            self.assertFalse(handoff.ready(12.))
            self.assertIn('alignment retained', handoff.status)
            for i in range(5):
                at = 12. + i*.08
                handoff.ingest(packet(i+10, at), at)
            self.assertTrue(handoff.ready(at))
            self.assertFalse(handoff.ingest(packet(14, at), at))
            self.assertTrue(handoff.ready(at))
            with self.assertRaisesRegex(ControlFault, 'robot moved'):
                handoff.try_validate(at, Mock(side_effect=ControlFault('robot moved during handoff')))
            self.assertFalse(handoff.ready(at+1.))

    def test_follow_keeps_zero_commands_past_handoff_deadline_then_switches_once(self):
        original = lifecycle.config
        def new_config(execute=False):
            args = original(execute)
            args.calibration_acquisition = 'consensus'
            args.handoff_wait = 'hold'
            args.handoff_timeout = 1.
            return args
        fixture = lifecycle.LiveAlignmentRuntimeTests()
        with patch.object(lifecycle, 'config', new_config):
            result = fixture.run_fake(done_after=1, stop_after=49,
                                      calibration_events={i: 0. for i in range(40, 48)})
        self.assertIsNone(result.caught)
        self.assertEqual(result.alignment.commit.call_count, 1)
        self.assertGreater(result.alignment.propose_hold.call_count, 30)
        self.assertGreater(result.session.send.call_count, 0)

    def test_new_cli_options_are_validated(self):
        args = options()
        self.assertEqual((args.calibration_acquisition, args.handoff_wait), ('consensus', 'hold'))
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(['--handoff_wait', 'resume'])

    def test_recorded_preview_and_handoff_pass_without_fictitious_base_velocity(self):
        data = json.loads((Path(__file__).parent/'fixtures/stationary_acquisition.json').read_text())
        args = options()
        vars(args).update(data['settings'])
        args.ur10e_output_correction = True  # Preserve this historical recording's target/base coordinates.
        engine = CooperativeController(args)
        def frame(row):
            return CalibrationFrame(row['sequence'], row['captured_at'],
                                    {n: np.array(p) for n, p in row['poses_camera_model'].items()},
                                    row['raw_mean_px'], row['raw_max_px'], row['valid_counts'])
        def states(raw):
            return {n: SimpleNamespace(joints=np.array(s['q']), joint_velocity=np.array(s['qd']),
                                      tcp=np.array(s['tcp']), speed=np.array(s['speed']), robot_timestamp=50.)
                    for n, s in raw.items()}
        def targets(bases, robots):
            leader = bases[engine.leader] @ flange_from_tcp(robots[engine.leader].tcp, engine.offsets[engine.leader])
            return {engine.follower: inverse(bases[engine.follower]) @ leader @ engine.path.pose(0.) @ engine.offsets[engine.follower]}
        initial = states(data['request']['initial_robots'])
        ready_at = None
        for row in data['preview']:
            engine.ingest(frame(row), row['at'], allow_reacquire=True)
            try:
                bases = engine.state(row['at'], False).base_poses
                check_alignment_request(args, engine.offsets, bases, initial, targets(bases, initial), engine.models, engine.guard)
            except ControlFault:
                continue
            ready_at = row['at']
            break
        self.assertIsNotNone(ready_at)
        self.assertLess(ready_at-data['preview'][0]['at'], 1.)
        engine.initial_tcp = initial[engine.follower].tcp
        started = data['endpoint']['at']
        handoff = CalibrationHandoff(engine.gate, started, consensus=True, wait_at_rest=True,
                                     estimator_factory=engine._new_estimator, mappings=engine.mappings, leader=engine.leader)
        endpoint = states(data['endpoint_robots'])
        old_bases = {n: np.array(p) for n, p in data['request']['initial_bases'].items()}
        nominal = {n: np.array(p) for n, p in data['request']['tcp_goals'].items()}
        passed = False
        for row in data['handoff']:
            handoff.ingest(frame(row), row['at'])
            if not handoff.ready(row['at']):
                continue
            snapshot = handoff.estimator.estimate(row['at'], leader=engine.leader)
            for twist in snapshot.base_twists.values():
                np.testing.assert_array_equal(twist, np.zeros(6))
            def validate():
                check_handoff_snapshot(args, old_bases, endpoint, snapshot.base_poses, endpoint,
                                       offsets=engine.offsets, nominal_targets=nominal,
                                       live_targets=targets(snapshot.base_poses, endpoint))
                engine.gate, engine.estimator = handoff.tracker, handoff.estimator
                engine.invalid_since = None
                proposal = engine.prepare(row['at'], endpoint, .04)
                self.assertTrue(np.isfinite(proposal.twists[engine.follower]).all())
            if handoff.try_validate(row['at'], validate):
                passed = True
                self.assertLess(row['at']-started, 2.)
                break
        self.assertTrue(passed, handoff.reason)


if __name__ == '__main__':
    unittest.main()

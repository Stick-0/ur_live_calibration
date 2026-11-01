"""Common mounting-plane invariants through control filters and persistence."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from urcal.control.geometry import calibration_output_mapping, inverse, model_from_controller_base, pose_matrix
from urcal.control.policy import CalibrationFrame
from urcal.cooperative.base_plane import constrain_base_snapshot, project_output_bases
from urcal.cooperative.controller import CooperativeController
from urcal.cooperative.state import RelativeSnapshot
from urcal.simulation import snapshot as storage
from urcal.simulation.runtime import SimulationSession

from test_cooperative_controller import settings
from test_simulation_snapshot import AcquisitionRig, config, initial_state
from test_simulation_runtime import options


ROBOTS = ("ur5e", "ur10e")


def mappings(corrected=True):
    return {name: calibration_output_mapping(name, ur10e_output_correction=corrected) for name in ROBOTS}


def bases():
    return {"ur5e": pose_matrix([-.45, -.1, 1.8, .2, -.3, .5]),
            "ur10e": pose_matrix([.5, .05, 1.9, .3, -.25, -.6])}


def snapshot(poses, twists=None):
    return RelativeSnapshot(10., deepcopy(poses),
        {name: np.zeros(6) for name in ROBOTS} if twists is None else deepcopy(twists),
        inverse(poses["ur10e"]) @ poses["ur5e"], np.zeros(6),
        {"ur5e": .1, "ur10e": .2}, {"ur5e": 9.9, "ur10e": 9.8},
        {"ur5e": 9.95, "ur10e": 9.96}, {"ur5e": .01, "ur10e": .02})


class CommonPlaneControlTests(unittest.TestCase):
    def assert_plane(self, output, selected):
        physical = {name: output[name] @ inverse(selected[name]) @ model_from_controller_base(name)
                    for name in ROBOTS}
        n = physical["ur5e"][:3, 2]
        np.testing.assert_allclose(n, physical["ur10e"][:3, 2], atol=1e-12)
        self.assertAlmostEqual(float(n @ (physical["ur10e"][:3, 3] - physical["ur5e"][:3, 3])),
                               0., places=12)
        return physical

    def test_both_output_conventions_use_physical_base_plane_and_preserve_metadata(self):
        for correction in (False, True):
            selected = mappings(correction)
            raw = snapshot(bases())
            original = deepcopy(raw)
            projected = constrain_base_snapshot(raw, "ur10e", selected)
            physical = self.assert_plane(projected.base_poses, selected)
            for field in ("at", "measurement_age", "measurement_times", "received_times", "prediction_dt"):
                self.assertEqual(getattr(projected, field), getattr(raw, field))
            for name in ROBOTS:
                np.testing.assert_array_equal(raw.base_poses[name], original.base_poses[name])
                np.testing.assert_array_equal(projected.base_twists[name], np.zeros(6))
            np.testing.assert_allclose(projected.relative_pose,
                inverse(projected.base_poses["ur10e"]) @ projected.base_poses["ur5e"], atol=1e-12)
            if not correction:
                normal = physical["ur5e"][:3, 2]
                difference = projected.base_poses["ur10e"][:3, 3] - projected.base_poses["ur5e"][:3, 3]
                self.assertAlmostEqual(float(normal @ difference), -.005, places=12)

    def test_velocity_projection_is_tangent_and_matches_first_order_plane_projection(self):
        for correction in (False, True):
            selected = mappings(correction)
            start = project_output_bases(bases(), selected)
            twists = {"ur5e": np.array([.04, -.01, .03, .2, -.1, .07]),
                      "ur10e": np.array([-.02, .03, -.01, -.1, .2, -.05])}
            projected = constrain_base_snapshot(snapshot(start, twists), "ur5e", selected)
            dt = 1e-7
            advanced = deepcopy(start)
            for name in ROBOTS:
                advanced[name][:3, 3] += twists[name][:3] * dt
                advanced[name][:3, :3] = Rotation.from_rotvec(twists[name][3:] * dt).as_matrix() @ start[name][:3, :3]
            future = project_output_bases(advanced, selected)
            for name in ROBOTS:
                actual = np.r_[(future[name][:3, 3] - start[name][:3, 3]) / dt,
                    Rotation.from_matrix(future[name][:3, :3] @ start[name][:3, :3].T).as_rotvec() / dt]
                np.testing.assert_allclose(projected.base_twists[name], actual, atol=2e-7)
            relative_future = inverse(future["ur5e"]) @ future["ur10e"]
            actual_relative = np.r_[(relative_future[:3, 3] - projected.relative_pose[:3, 3]) / dt,
                Rotation.from_matrix(relative_future[:3, :3] @ projected.relative_pose[:3, :3].T).as_rotvec() / dt]
            np.testing.assert_allclose(projected.relative_twist, actual_relative, atol=2e-7)

    def test_common_rigid_camera_velocity_remains_zero_relative_motion(self):
        poses = bases()
        angular, translation = np.array([.12, -.08, .17]), np.array([.02, -.01, .03])
        velocities = {name: np.r_[translation + np.cross(angular, pose[:3, 3]), angular]
                      for name, pose in poses.items()}
        result = constrain_base_snapshot(snapshot(poses, velocities), "ur10e", mappings(False))
        np.testing.assert_allclose(result.relative_twist, np.zeros(6), atol=1e-12)
        for name in ROBOTS:
            np.testing.assert_allclose(result.base_twists[name],
                np.r_[translation + np.cross(angular, result.base_poses[name][:3, 3]), angular], atol=1e-12)

    def test_controller_projects_preview_and_both_sides_of_temporal_filters(self):
        args = settings(common_base_plane=True, base_motion_mode="pose", base_pose_window=5,
                        leader_response="bounded", live_calibration_recovery=True)
        engine = CooperativeController(args)
        raw = bases()
        frame = CalibrationFrame(1, 10., raw, {name: 1. for name in ROBOTS},
                                  {name: 2. for name in ROBOTS}, {name: 7 for name in ROBOTS})
        self.assertTrue(engine.ingest(frame, 10.01))
        preview = engine.state(10.01, active=False)
        self.assert_plane(preview.base_poses, engine.mappings)
        input_to_smoother = []
        original_apply = engine.pose_smoother.apply
        def smooth(value, now, leader):
            input_to_smoother.append(value)
            return original_apply(value, now, leader)
        engine.pose_smoother.apply = smooth
        # A recovery blend can create noncoplanar intermediate coordinates.
        def blend(value, now, leader):
            changed = deepcopy(value.base_poses)
            changed["ur10e"][2, 3] += .01
            changed["ur10e"][:3, :3] = Rotation.from_rotvec([.01, 0, 0]).as_matrix() @ changed["ur10e"][:3, :3]
            return replace(value, base_poses=changed)
        engine.live_calibration.blend = blend
        final = engine.state(10.02, active=True)
        self.assert_plane(input_to_smoother[0].base_poses, engine.mappings)
        self.assert_plane(final.base_poses, engine.mappings)
        self.assertEqual(final.measurement_times, preview.measurement_times)

    def test_legacy_args_do_not_enable_new_constraint_and_boolean_is_strict(self):
        args = settings()
        del args.common_base_plane
        self.assertFalse(CooperativeController(args).common_base_plane)
        for invalid in (1, "true", None):
            args.common_base_plane = invalid
            with self.assertRaisesRegex(ValueError, "boolean"):
                CooperativeController(args)


class CommonPlaneSnapshotTests(unittest.TestCase):
    assert_plane = CommonPlaneControlTests.assert_plane

    def test_tilted_simulated_base_motion_keeps_truth_and_virtual_sensor_on_same_plane(self):
        for corrected in (False, True):
            for motion in ("bases", "both"):
                with self.subTest(output_correction=corrected, motion=motion):
                    selected = mappings(corrected)
                    captured = initial_state()
                    captured["calibration"].update(common_base_plane=True,
                                                   ur10e_output_correction=corrected)
                    captured["bases"] = project_output_bases(bases(), selected)
                    captured["calibration"]["poses"] = {
                        name: captured["bases"][name] @ inverse(selected[name]) for name in ROBOTS}
                    args = options("follow", motion, "--base_motion_mode", "pose", "--filter_alpha", "1",
                                   "--stable_frames", "1", "--base_pose_window", "0")
                    session = SimulationSession(args, captured)
                    physical = self.assert_plane(session.plant.bases, selected)
                    normal = physical["ur5e"][:3, 2]
                    self.assertGreater(abs(normal[1]), .01)  # Camera Y would leave this plane.
                    for step in (50, 150, 250, 450):
                        session.steps = step
                        moved, _ = session._external_motion()
                        self.assert_plane(moved, selected)
                        delta = moved["ur5e"][:3, 3] - captured["bases"]["ur5e"][:3, 3]
                        self.assertGreater(np.linalg.norm(delta), 0.)
                        self.assertAlmostEqual(float(normal @ delta), 0., places=12)
                        np.testing.assert_allclose(moved["ur10e"][:3, 3]
                            - captured["bases"]["ur10e"][:3, 3], -delta, atol=1e-12)
                        session.plant.bases = moved
                        sensor = session._calibration()
                        session._ingest(session.now)
                        estimate = session.engine.state(session.now, active=False)
                        for name in ROBOTS:
                            np.testing.assert_allclose(sensor[name] @ selected[name], moved[name], atol=1e-12)
                            np.testing.assert_allclose(estimate.base_poses[name], moved[name], atol=1e-12)

    def test_legacy_simulated_base_motion_retains_camera_y_direction(self):
        for motion in ("bases", "both"):
            session = SimulationSession(options("follow", motion), initial_state())
            self.assertFalse(session.args.common_base_plane)
            session.steps = 250
            moved, _ = session._external_motion()
            first = moved["ur5e"][:3, 3] - session.initial["bases"]["ur5e"][:3, 3]
            np.testing.assert_allclose(first[[0, 2]], 0., atol=1e-12)
            self.assertGreater(first[1], 0.)
            np.testing.assert_allclose(moved["ur10e"][:3, 3]
                - session.initial["bases"]["ur10e"][:3, 3], -first, atol=1e-12)

    def test_capture_records_plane_and_projects_filtered_model_poses_consistently(self):
        for corrected in (False, True):
            rig, args = AcquisitionRig(), config()
            args.common_base_plane = True
            args.ur10e_output_correction = corrected
            with rig.patches():
                captured = storage.capture_initial_state(args)
            self.assertTrue(captured["calibration"]["common_base_plane"])
            selected = mappings(corrected)
            self.assert_plane(captured["bases"], selected)
            for name in ROBOTS:
                np.testing.assert_allclose(captured["bases"][name],
                    captured["calibration"]["poses"][name] @ selected[name], atol=1e-12)
            rig.assert_closed(self)

    def test_saved_flag_roundtrip_and_legacy_snapshot_are_not_reinterpreted(self):
        saved = initial_state()
        legacy = storage.validate_initial_state(saved)
        self.assertFalse(legacy["calibration"]["common_base_plane"])
        selected = mappings()
        saved["bases"] = project_output_bases(saved["bases"], selected)
        saved["calibration"]["poses"] = {name: saved["bases"][name] @ inverse(selected[name]) for name in ROBOTS}
        saved["calibration"]["common_base_plane"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = storage.save_initial_state(Path(directory) / "capture.json", saved)
            loaded = storage.load_initial_state(path)
        self.assertTrue(loaded["calibration"]["common_base_plane"])
        self.assert_plane(loaded["bases"], selected)
        args = options()
        args.common_base_plane = False
        session = SimulationSession(args, loaded)
        self.assertTrue(session.engine.common_base_plane)
        args.common_base_plane = True
        session = SimulationSession(args, legacy)
        self.assertFalse(session.engine.common_base_plane)

    def test_snapshot_rejects_false_plane_claim_and_nonboolean_flag(self):
        saved = initial_state()
        saved["calibration"]["common_base_plane"] = True
        with self.assertRaisesRegex(ValueError, "mounting plane"):
            storage.validate_initial_state(saved)
        for invalid in (1, "false", None):
            saved["calibration"]["common_base_plane"] = invalid
            with self.assertRaisesRegex(ValueError, "boolean"):
                storage.validate_initial_state(saved)


if __name__ == "__main__":
    unittest.main()

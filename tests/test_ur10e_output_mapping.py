"""Optional PnP output mapping must not alter native encoder kinematics."""

import unittest

import numpy as np

from urcal.control.geometry import calibration_output_mapping, model_from_controller_base, pose_matrix
from urcal.control.policy import CalibrationFrame
from urcal.control.runtime import parse_args
from urcal.cooperative.controller import CooperativeController


def engine(corrected=None):
    arguments = ["--ur5e_ip", "192.0.2.5", "--ur10e_ip", "192.0.2.10",
                 "--stable_frames", "1", "--filter_alpha", "1", "--no_common_base_plane"]
    if corrected is not None:
        arguments.append("--ur10e_output_correction" if corrected else "--no_ur10e_output_correction")
    return CooperativeController(parse_args(arguments))


class UR10eOutputMappingTests(unittest.TestCase):
    def test_explicit_opt_out_uses_raw_pnp_output_for_both_bases(self):
        controller = engine(False)
        self.assertFalse(controller.args.ur10e_output_correction)
        poses = {"ur5e": pose_matrix([-.3, .2, 2., .1, -.2, .3]),
                 "ur10e": pose_matrix([.5, .3, 2.1, -.2, .1, -.4])}
        frame = CalibrationFrame(1, 10., poses, {name: 1. for name in poses},
                                 {name: 2. for name in poses}, {name: 7 for name in poses})
        self.assertTrue(controller.ingest(frame, 10.1))
        state = controller.state(10.1, active=False)
        for name in poses:
            np.testing.assert_array_equal(controller.mappings[name], np.eye(4))
            np.testing.assert_array_equal(state.base_poses[name], poses[name])

    def test_default_corrects_output_in_local_axes_and_never_changes_native_fk_or_jacobian(self):
        raw, corrected = engine(False), engine()
        self.assertTrue(corrected.args.ur10e_output_correction)
        self.assertTrue(engine(True).args.ur10e_output_correction)
        nominal = model_from_controller_base("ur10e")
        np.testing.assert_array_equal(calibration_output_mapping("ur10e"), nominal)
        np.testing.assert_array_equal(nominal[:3, :3], np.diag([-1., -1., 1.]))
        np.testing.assert_array_equal(nominal[:3, 3], [0., 0., .005])
        poses = {"ur5e": np.eye(4), "ur10e": pose_matrix([.5, .3, 2., .4, -.3, .6])}
        frame = CalibrationFrame(1, 10., poses, {name: 1. for name in poses},
                                 {name: 2. for name in poses}, {name: 7 for name in poses})
        corrected.ingest(frame, 10.1)
        state = corrected.state(10.1, active=False)
        np.testing.assert_allclose(state.base_poses["ur10e"], poses["ur10e"] @ nominal, atol=1e-12)
        np.testing.assert_allclose(state.base_poses["ur10e"][:3, 3] - poses["ur10e"][:3, 3],
                                   .005 * poses["ur10e"][:3, 2], atol=1e-12)
        for robot in poses:
            for q in (np.zeros(6), np.array([.3, -1., 1.5, -.7, 1.2, -.6])):
                raw_pose, raw_jacobian = raw.models[robot].pose_and_jacobian(q)
                corrected_pose, corrected_jacobian = corrected.models[robot].pose_and_jacobian(q)
                np.testing.assert_array_equal(raw_pose, corrected_pose)
                np.testing.assert_array_equal(raw_jacobian, corrected_jacobian)
                np.testing.assert_array_equal(raw.models[robot].points(q), corrected.models[robot].points(q))

    def test_output_mapping_validates_robot_and_boolean_and_never_changes_ur5e(self):
        for correction in (False, True):
            np.testing.assert_array_equal(calibration_output_mapping(
                "ur5e", ur10e_output_correction=correction), np.eye(4))
        with self.assertRaises(ValueError):
            calibration_output_mapping("ur3e")
        for invalid in ("false", 0, None):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                calibration_output_mapping("ur10e", ur10e_output_correction=invalid)

    def test_old_saved_research_arguments_keep_their_explicit_frame_interpretation(self):
        args = engine().args
        del args.ur10e_output_correction
        legacy = CooperativeController(args)
        np.testing.assert_array_equal(legacy.mappings["ur10e"], model_from_controller_base("ur10e"))


if __name__ == "__main__":
    unittest.main()

"""Package relocation checks: process identity, assets, launchers and isolation."""

import ast
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from urcal.control.policy import CalibrationFrame


BUNDLE = Path(__file__).resolve().parents[1]
LEGACY_MODULES = {
    "infer_live", "infer_live_ur10e", "infer_live_dual", "infer_live_10", "network3", "vote2", "my_pnp",
    "control_dual", "dual_control_geometry", "dual_control_policy", "dual_control_vision", "dual_control_gui",
    "control_cooperative", "relative_state", "cooperative_controller", "cooperative_task",
    "cooperative_kinematics", "cooperative_logging", "analyze_cooperative_log", "simulate_cooperative",
}


def _inspect_spawned_objects(frame, vision_target, gui_target, connection):
    """Receive real spawn-pickled values; never invoke either hardware worker."""
    try:
        from urcal.control.policy import CalibrationFrame as ChildFrame
        from urcal.control.vision import CalibrationFrame as VisionFrame, vision_worker
        from urcal.control.gui import gui_worker
        connection.send({"frame_identity": type(frame) is ChildFrame is VisionFrame,
                         "frame_module": type(frame).__module__, "sequence": frame.sequence,
                         "pose": frame.poses["ur5e"].tolist(),
                         "vision_identity": vision_target is vision_worker,
                         "gui_identity": gui_target is gui_worker,
                         "hardware_modules": sorted({"rtde_control", "rtde_receive", "pyrealsense2"} & set(sys.modules))})
    except BaseException as exc:
        connection.send({"error": repr(exc)})
    finally:
        connection.close()


class PackageLayoutTests(unittest.TestCase):
    def run_isolated(self, source, cwd=None):
        env = dict(os.environ)
        env["PYTHONPATH"] = str(BUNDLE) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        result = subprocess.run([sys.executable, "-B", "-c", source], cwd=cwd or BUNDLE,
                                env=env, capture_output=True, text=True, timeout=30.)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def test_production_imports_do_not_fall_back_to_legacy_top_level_modules(self):
        invalid = []
        for path in (BUNDLE / "urcal").rglob("*.py"):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                modules = ([node.module] if isinstance(node, ast.ImportFrom) and node.module
                           else [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
                for module in modules:
                    if module.split(".")[0] in LEGACY_MODULES:
                        invalid.append(f"{path.relative_to(BUNDLE)}:{node.lineno}: {module}")
        self.assertEqual(invalid, [])

    def test_reference_and_packaged_inference_keep_distinct_dependencies(self):
        self.run_isolated("""
import importlib.util
import sys
from urcal.paths import BUNDLE
from urcal.inference import infer_live as modern
sys.path.insert(0, str(BUNDLE / 'reference'))
spec = importlib.util.spec_from_file_location('isolated_original_live', BUNDLE / 'reference' / 'infer_live.py')
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
for attribute, module in [('EMPnP', 'my_pnp'), ('RobotJointDetection', 'network3'), ('PVNetVoteConfig', 'vote2')]:
    baseline = getattr(original, attribute)
    optimized = getattr(modern, attribute)
    assert baseline is not optimized, attribute
    assert baseline.__module__ == module, baseline.__module__
    assert optimized.__module__ == 'urcal.inference.' + module, optimized.__module__
from urcal.inference import infer_live_ur10e, infer_live_dual
assert infer_live_ur10e.RobotJointDetection is modern.RobotJointDetection
assert infer_live_dual.RobotJointDetection is modern.RobotJointDetection
assert not {'rtde_control', 'rtde_receive', 'pyrealsense2'} & set(sys.modules)
""")

    def test_offline_package_import_does_not_load_hardware_or_neural_runtime(self):
        self.run_isolated("""
import sys
import urcal
import urcal.tools.analyze_log
import urcal.tools.simulate
import urcal.control.runtime
import urcal.cooperative.runtime
import urcal.simulation.runtime
unexpected = {'rtde_control', 'rtde_receive', 'pyrealsense2', 'torch', 'tkinter'} & set(sys.modules)
assert not unexpected, unexpected
""")

    def test_shared_asset_root_stays_at_bundle_from_another_working_directory(self):
        with tempfile.TemporaryDirectory() as working:
            output = self.run_isolated("""
import json
from pathlib import Path
from urcal.paths import BUNDLE
from urcal.control import runtime
from urcal.inference import infer_live_dual as dual
config = runtime.parse_args(['--mode', 'follow', '--ur5e_ip', 'test5', '--ur10e_ip', 'test10'])
assert runtime.BUNDLE == dual.BUNDLE == BUNDLE
assert config.ur5e_checkpoint == BUNDLE / 'trained_models' / 'v3m2a3' / 'best.pt'
assert config.ur10e_checkpoint == BUNDLE / 'trained_models' / 'ur10e_2' / 'best.pt'
for asset in ['docs/Research_Implementation_Plan.txt', 'configs/workspace.example.json',
              'torch_cache/hub/checkpoints/resnet18-f37072fd.pth']:
    assert (BUNDLE / asset).is_file(), asset
print(json.dumps({'cwd': str(Path.cwd()), 'bundle': str(BUNDLE)}))
""", cwd=working)
            result = json.loads(output)
            self.assertEqual(result["cwd"], working)
            self.assertEqual(Path(result["bundle"]), BUNDLE)

    def test_shell_entrypoints_preserve_caller_paths_and_reference_selection(self):
        launchers = {
            "run_ur5e.sh": "urcal.inference.infer_live",
            "run_ur10e.sh": "urcal.inference.infer_live_ur10e",
            "run_dual.sh": "urcal.inference.infer_live_dual",
            "run_dual_control.sh": "urcal.control.runtime",
            "run_cooperative.sh": "urcal.cooperative.runtime",
            "run_simulation.sh": "urcal.simulation.runtime",
        }
        with tempfile.TemporaryDirectory() as working:
            interpreter = Path(working) / "capture_interpreter"
            interpreter.write_text(f"#!{sys.executable}\n" +
                                   "import json, os, sys\n" +
                                   "print(json.dumps({'cwd':os.getcwd(),'argv':sys.argv[1:],"
                                   "'pythonpath':os.environ['PYTHONPATH'],'torch_home':os.environ['TORCH_HOME']}))\n")
            interpreter.chmod(0o755)
            env = dict(os.environ, ROBOTPOSE_PYTHON=str(interpreter), ROBOTPOSE_REFERENCE="0", PYTHONPATH="/existing/python/path")
            arguments = ["--custom-path", "relative directory/input.json"]
            for launcher, module in launchers.items():
                with self.subTest(launcher=launcher):
                    result = subprocess.run([str(BUNDLE / launcher), *arguments], cwd=working, env=env,
                                            capture_output=True, text=True, timeout=10.)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    capture = json.loads(result.stdout)
                    self.assertEqual(capture["cwd"], working)
                    self.assertEqual(capture["argv"][:2], ["-m", module])
                    self.assertEqual(capture["argv"][-2:], arguments)
                    self.assertEqual(capture["pythonpath"], str(BUNDLE) + os.pathsep + "/existing/python/path")
                    self.assertEqual(Path(capture["torch_home"]), BUNDLE / "torch_cache")
            env["ROBOTPOSE_REFERENCE"] = "1"
            for launcher, script in (("run_ur5e.sh", "infer_live.py"), ("run_ur10e.sh", "infer_live_ur10e.py")):
                with self.subTest(reference=launcher):
                    result = subprocess.run([str(BUNDLE / launcher), *arguments], cwd=working, env=env,
                                            capture_output=True, text=True, timeout=10.)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    capture = json.loads(result.stdout)
                    self.assertEqual(Path(capture["argv"][0]), BUNDLE / "reference" / script)
                    self.assertEqual(capture["argv"][-2:], arguments)
                    self.assertEqual(capture["cwd"], working)

    def test_real_spawn_preserves_canonical_frame_and_worker_identities(self):
        from urcal.control.vision import vision_worker
        from urcal.control.gui import gui_worker
        frame = CalibrationFrame(7, 10., {"ur5e": np.eye(4), "ur10e": np.eye(4)},
                                 {name: 1. for name in ("ur5e", "ur10e")},
                                 {name: 2. for name in ("ur5e", "ur10e")},
                                 {name: 7 for name in ("ur5e", "ur10e")})
        context = mp.get_context("spawn")
        receiving, sending = context.Pipe(duplex=False)
        child = context.Process(target=_inspect_spawned_objects, args=(frame, vision_worker, gui_worker, sending))
        try:
            child.start()
            sending.close()
            self.assertTrue(receiving.poll(15.), "spawned child did not return")
            result = receiving.recv()
            child.join(timeout=5.)
            self.assertEqual(child.exitcode, 0)
            self.assertNotIn("error", result)
            self.assertTrue(result["frame_identity"])
            self.assertEqual(result["frame_module"], "urcal.control.policy")
            self.assertEqual(result["sequence"], 7)
            np.testing.assert_array_equal(result["pose"], np.eye(4))
            self.assertTrue(result["vision_identity"])
            self.assertTrue(result["gui_identity"])
            self.assertEqual(result["hardware_modules"], [])
        finally:
            if child.pid is not None and child.is_alive():
                child.terminate()
                child.join(timeout=5.)
            receiving.close()
            sending.close()


if __name__ == "__main__":
    unittest.main()

"""Qt font bootstrap checks without opening windows or importing hardware."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from urcal.inference.display import configure_opencv_fonts, import_opencv


class OpenCVFontTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.fonts = Path(self.directory.name) / "fonts"
        self.fonts.mkdir()
        (self.fonts / "example.ttf").write_bytes(b"font")
        self.missing = str(Path(self.directory.name) / "missing")

    def test_missing_wheel_directory_uses_installed_fonts(self):
        with patch.dict(os.environ, {"QT_QPA_FONTDIR": self.missing}):
            result = configure_opencv_fonts(candidates=[self.fonts])
            self.assertEqual(result, str(self.fonts))
            self.assertEqual(os.environ["QT_QPA_FONTDIR"], str(self.fonts))

    def test_valid_existing_user_directory_is_unchanged(self):
        with patch.dict(os.environ, {"QT_QPA_FONTDIR": str(self.fonts)}):
            self.assertEqual(configure_opencv_fonts(candidates=[]), str(self.fonts))

    def test_import_restores_user_preference_overwritten_by_wheel(self):
        module = object()

        def bootstrap(name):
            self.assertEqual(name, "cv2")
            os.environ["QT_QPA_FONTDIR"] = self.missing
            return module

        with patch.dict(os.environ, {"QT_QPA_FONTDIR": str(self.fonts)}), \
                patch("urcal.inference.display.importlib.import_module", side_effect=bootstrap):
            self.assertIs(import_opencv(), module)
            self.assertEqual(os.environ["QT_QPA_FONTDIR"], str(self.fonts))

    def test_unset_variable_is_not_added_for_headless_builds(self):
        with patch.dict(os.environ):
            os.environ.pop("QT_QPA_FONTDIR", None)
            self.assertIsNone(configure_opencv_fonts(candidates=[self.fonts]))
            self.assertNotIn("QT_QPA_FONTDIR", os.environ)

    def test_fallback_requires_font_files_and_skips_empty_directories(self):
        empty = Path(self.directory.name) / "empty"
        empty.mkdir()
        with patch.dict(os.environ, {"QT_QPA_FONTDIR": self.missing}):
            self.assertEqual(configure_opencv_fonts(candidates=[empty]), self.missing)
            self.assertEqual(configure_opencv_fonts(candidates=[empty, self.fonts]), str(self.fonts))

    def test_repeated_configuration_keeps_working_directory(self):
        with patch.dict(os.environ, {"QT_QPA_FONTDIR": self.missing}):
            configure_opencv_fonts(candidates=[self.fonts])
            self.assertEqual(configure_opencv_fonts(candidates=[]), str(self.fonts))


if __name__ == "__main__":
    unittest.main()

"""Import OpenCV without leaving its optional Qt font directory broken."""

import importlib
import os
from pathlib import Path


FONT_DIRECTORIES = (
    "/usr/share/fonts/truetype/dejavu",
    "/usr/share/fonts/truetype/liberation2",
    "/usr/share/fonts/truetype/liberation",
    "/usr/local/share/fonts",
)


def _directory(value):
    try:
        return bool(value) and Path(value).is_dir()
    except (OSError, ValueError):
        return False


def configure_opencv_fonts(preferred=None, candidates=FONT_DIRECTORIES):
    """Repair a missing Qt font path, preserving a valid explicit preference.

    Some OpenCV wheels set QT_QPA_FONTDIR to a directory absent from the wheel.
    Choose an existing directory containing font files before the first window.
    Headless/non-Qt builds that leave this variable unset require no change.
    """
    current = os.environ.get("QT_QPA_FONTDIR")
    if _directory(preferred):
        os.environ["QT_QPA_FONTDIR"] = str(preferred)
        return str(preferred)
    if current is None or _directory(current):
        return current
    for candidate in candidates:
        path = Path(candidate)
        try:
            has_fonts = path.is_dir() and any(
                entry.is_file() and entry.suffix.lower() in {".ttf", ".otf", ".ttc"}
                for entry in path.iterdir()
            )
        except OSError:
            continue
        if has_fonts:
            os.environ["QT_QPA_FONTDIR"] = str(path)
            return str(path)
    return current


def import_opencv():
    """Keep user font settings even when cv2's bootstrap overwrites them."""
    preferred = os.environ.get("QT_QPA_FONTDIR")
    cv2 = importlib.import_module("cv2")
    configure_opencv_fonts(preferred)
    return cv2

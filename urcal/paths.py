"""Shared asset root, independent of the caller working directory."""

from pathlib import Path

BUNDLE = Path(__file__).resolve().parents[1]

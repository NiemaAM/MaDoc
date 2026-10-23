"""Puts src/ on the import path so the tests can import the stages (parser, converter, ...)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
"""Puts src/ on the import path so the tests can import the stages (parser, converter, ...), and empties the
caches of the models' answers between tests (each test has its own fake model)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(autouse=True)
def no_cached_answers():
    for module, cache in (("translator", "_answers"), ("classificator", "_results"),
                          ("spelling_checker", "_line_answers")):
        if module in sys.modules:
            getattr(sys.modules[module], cache).clear()
    yield


@pytest.fixture(autouse=True)
def kraken_command_line(monkeypatch):
    """The tests stand in for the kraken command line (subprocess.run): kraken's library, installed in the image
    of the app, is not used by them (kraken_engine.py has its own tests)."""
    monkeypatch.setenv("MADOC_KRAKEN_CLI", "1")
"""Shared test helpers.

The example folders use hyphens (e.g. ``prompt-template``), which are not valid
Python package names, so we load each ``example.py`` directly from its path with
importlib. ``load_example("prompt-template")`` returns the imported module.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parent.parent


def load_example(folder: str) -> ModuleType:
    """Import the ``example.py`` inside ``folder`` and return the module."""
    path = ROOT / folder / "example.py"
    module_name = f"example_{folder.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec and spec.loader, f"could not load spec for {path}"
    module = importlib.util.module_from_spec(spec)
    # Register before executing so dataclass annotation resolution (which looks
    # the module up in sys.modules by name) works under Python 3.12+.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

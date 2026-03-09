"""
OVEP suite conftest: injects OV Python package directories into sys.path
at collection time and skips the entire OVEP suite for Debug builds.
"""

import os
import sys
import pytest
from pathlib import Path


def _inject_ov_python_paths() -> None:
    ov_dir = os.environ.get("OV_INSTALL_DIR", "")
    if not ov_dir:
        return
    for sub in ["python", os.path.join("python", "python3")]:
        p = str(Path(ov_dir) / sub)
        if p not in sys.path:
            sys.path.insert(0, p)


_inject_ov_python_paths()


def pytest_collection_modifyitems(items):
    if os.environ.get("BUILD_TYPE", "Release") == "Debug":
        skip = pytest.mark.skip(
            reason="OVEP requires OV Python bindings, which are not built for Debug configurations"
        )
        for item in items:
            if "tests" + os.sep + "ovep" in str(item.fspath):
                item.add_marker(skip)

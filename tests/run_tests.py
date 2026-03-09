#!/usr/bin/env python
"""
Test runner: configures the runtime environment per build type and launches
pytest against the requested test suite (ov / ort / ovep / all).

Usage:
    python run_tests.py --base-dir C:\\ov_build --build-type Release
    python run_tests.py --base-dir C:\\ov_build --build-type RelWithDebInfo --suite ovep
    python run_tests.py --base-dir C:\\ov_build --build-type Release --ov-device GPU --ort-device GPU -v
    python run_tests.py --base-dir C:\\ov_build --build-type Debug --suite ort

Exit code mirrors pytest: 0 = all pass, 1 = some failures, 5 = no tests collected.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).parent


# ── environment helpers ───────────────────────────────────────────────────────

def _ov_lib_paths(ov_install: Path, build_type: str) -> list:
    """Return OV runtime DLL directories needed on PATH."""
    intel64 = ov_install / "runtime" / "bin" / "intel64"
    paths, seen = [], set()
    for candidate in [build_type, "Release", "Debug"]:
        p = intel64 / candidate
        if str(p) not in seen:
            paths.append(str(p))
            seen.add(str(p))
    for tbb in [
        ov_install / "runtime" / "3rdparty" / "tbb" / "redist" / "intel64" / "vc14",
        ov_install / "runtime" / "3rdparty" / "tbb" / "bin" / "intel64" / "vc14",
        ov_install / "runtime" / "3rdparty" / "tbb" / "bin",
    ]:
        if tbb.is_dir():
            paths.append(str(tbb))
            break
    return paths


def build_subprocess_env(
    base_dir: str, build_type: str, ov_device: str, ort_device: str
) -> dict:
    """Return an os.environ copy augmented with OV paths and test variables."""
    env = os.environ.copy()
    env["BUILD_TYPE"] = build_type
    env["OV_DEVICE"]  = ov_device
    env["ORT_DEVICE"] = ort_device

    ov_install = Path(base_dir) / "install" / "openvino" / build_type
    env["OV_INSTALL_DIR"] = str(ov_install)

    if not ov_install.is_dir():
        print(f"  WARNING: OV install directory not found: {ov_install}")
        print("           OV and OVEP tests will be skipped.")
        return env

    if build_type == "Debug":
        # Python bindings are not built for Debug — skip OV path injection
        return env

    # PYTHONPATH — OV Python package
    py_dir  = str(ov_install / "python")
    py3_dir = str(ov_install / "python" / "python3")
    existing_py = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(p for p in [py_dir, py3_dir, existing_py] if p)

    # PATH + OPENVINO_LIB_PATHS — OV runtime DLLs
    lib_paths = _ov_lib_paths(ov_install, build_type)
    env["OPENVINO_LIB_PATHS"] = os.pathsep.join(lib_paths)
    env["PATH"] = os.pathsep.join(lib_paths) + os.pathsep + env.get("PATH", "")

    return env


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="OV / ORT / OVEP inference verification test runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--base-dir", required=True,
        help="Base build directory (e.g. C:\\ov_build)",
    )
    ap.add_argument(
        "--build-type", default="Release",
        choices=["Release", "RelWithDebInfo", "Debug"],
        help="CMake build configuration to test (default: Release)",
    )
    ap.add_argument(
        "--suite", default="all",
        choices=["all", "ov", "ort", "ovep"],
        help="Test suite to run: ov | ort | ovep | all  (default: all)",
    )
    ap.add_argument(
        "--ov-device", default="CPU", choices=["CPU", "GPU"],
        help="OpenVINO device for native OV and OVEP tests (default: CPU)",
    )
    ap.add_argument(
        "--ort-device", default="CPU", choices=["CPU", "GPU"],
        help="Device forwarded to ORT OpenVINO EP (default: CPU)",
    )
    ap.add_argument(
        "--verbose", "-v", action="store_true",
        help="Pass -v to pytest for verbose per-test output",
    )
    args = ap.parse_args()

    # Debug builds have no Python bindings → only ORT-CPU is valid
    suite = args.suite
    if args.build_type == "Debug" and suite in ("ov", "ovep"):
        print(
            "ERROR: OpenVINO Python bindings are not built in Debug mode.\n"
            "       Use --suite ort for Debug, or choose Release / RelWithDebInfo."
        )
        sys.exit(1)
    if args.build_type == "Debug" and suite == "all":
        print(
            "NOTE: OpenVINO Python bindings are disabled for Debug builds.\n"
            "      Running ORT (CPU) suite only. Use --build-type Release for OV and OVEP tests."
        )
        suite = "ort"

    suite_dirs = {
        "ov":   [str(TESTS_DIR / "ov")],
        "ort":  [str(TESTS_DIR / "ort")],
        "ovep": [str(TESTS_DIR / "ovep")],
        "all":  [
            str(TESTS_DIR / "ov"),
            str(TESTS_DIR / "ort"),
            str(TESTS_DIR / "ovep"),
        ],
    }
    test_paths = suite_dirs[suite]

    env = build_subprocess_env(args.base_dir, args.build_type, args.ov_device, args.ort_device)

    cmd = [sys.executable, "-m", "pytest"] + test_paths
    if args.verbose:
        cmd.append("-v")
    cmd += ["--tb=short", "-p", "no:cacheprovider"]

    ov_install = Path(args.base_dir) / "install" / "openvino" / args.build_type
    print("=" * 64)
    print(f"  Suite      : {suite.upper()}")
    print(f"  Build type : {args.build_type}")
    print(f"  OV device  : {args.ov_device}")
    print(f"  ORT device : {args.ort_device}")
    print(f"  OV install : {ov_install}")
    print(f"  Command    : {' '.join(cmd)}")
    print("=" * 64 + "\n")

    result = subprocess.run(cmd, env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()

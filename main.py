"""
OpenVINO Multi-Agent Build System - Entry Point

Usage:
    python main.py <base_directory> [build_types...] [--ort-only]

Examples:
    python main.py C:\\ov_build                              # All 3 variants, full pipeline
    python main.py C:\\ov_build Release                       # Release only
    python main.py C:\\ov_build Release Debug                 # Release + Debug
    python main.py D:\\projects\\ov Release RelWithDebInfo    # Custom path
    python main.py C:\\ov_build --ort-only                    # ORT phase only (OV already built)
    python main.py C:\\ov_build --verify-ov-only               # OV verification only
"""

import sys
import os
import subprocess
import logging
from pathlib import Path

# ── Venv bootstrap ────────────────────────────────────────────────────────────
# If we are not already running inside the project venv, create it (if needed),
# install all requirements, and re-execute this script with the venv Python.
# This ensures CMake always receives a Python interpreter that has 'packaging',
# 'numpy', 'onnx', etc. available.

PROJECT_ROOT = Path(__file__).parent
VENV_DIR = PROJECT_ROOT / ".venv"
VENV_PYTHON = VENV_DIR / "Scripts" / "python.exe"   # Windows
REQUIREMENTS = PROJECT_ROOT / "requirements.txt"


def _bootstrap_venv():
    """Create venv + install deps (only when needed), then re-exec with the venv Python."""
    # 1. Create venv if it does not exist yet
    fresh_venv = not VENV_PYTHON.exists()
    if fresh_venv:
        print(f"[bootstrap] Creating virtual environment at {VENV_DIR} ...")
        subprocess.check_call([sys.executable, "-m", "venv", str(VENV_DIR)])
        print("[bootstrap] Virtual environment created.")

    # 2. Install requirements only when:
    #    - venv was just created (fresh install), OR
    #    - requirements.txt is newer than the venv sentinel file (.venv/.installed)
    sentinel = VENV_DIR / ".installed"
    needs_install = fresh_venv or not sentinel.exists() or (
        REQUIREMENTS.stat().st_mtime > sentinel.stat().st_mtime
    )

    if needs_install:
        print(f"[bootstrap] Installing requirements from {REQUIREMENTS} ...")
        raw_proxy = (os.getenv("http_proxy") or os.getenv("HTTP_PROXY") or "")
        # Sanitize double-colon typo: http:// -> http://
        proxy = raw_proxy.replace("http:://", "http://").replace("https:://", "https://")
        pip_cmd = [str(VENV_PYTHON), "-m", "pip", "install", "--upgrade",
                   "-r", str(REQUIREMENTS)]
        if proxy:
            pip_cmd += ["--proxy", proxy,
                        "--trusted-host", "pypi.org",
                        "--trusted-host", "files.pythonhosted.org"]
        result = subprocess.run(pip_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("[bootstrap] WARNING: pip install had errors (continuing anyway):")
            print(result.stderr[-2000:])
        else:
            print("[bootstrap] Requirements installed.")
            sentinel.touch()  # update timestamp so next run skips install
    else:
        print("[bootstrap] Venv up-to-date, skipping pip install.")

    # 3. Re-execute this script with the venv Python
    print(f"[bootstrap] Re-launching with venv Python: {VENV_PYTHON}")
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON)] + sys.argv)
    # os.execv replaces the current process — nothing below this line runs.


# Guard: only bootstrap when NOT already inside the venv
if sys.executable.lower() != str(VENV_PYTHON).lower():
    _bootstrap_venv()
# ── End bootstrap ─────────────────────────────────────────────────────────────

# Add project root to Python path so imports work
sys.path.insert(0, str(PROJECT_ROOT))

from config import BuildConfig
from orchestrator import Orchestrator


def setup_logging(base_dir: str):
    """Log to both console and file."""
    log_dir = Path(base_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = str(log_dir / "build_agent.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)-20s] %(levelname)-7s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
        ],
    )


def main():
    # ── Parse arguments ──
    if len(sys.argv) < 2:
        print(__doc__)
        print("ERROR: Please provide a base directory.\n")
        print("Quick start:")
        print('  python main.py C:\\ov_build')
        sys.exit(1)

    base_dir = sys.argv[1]

    # Optional: specify which build types and/or --ort-only flag
    remaining = sys.argv[2:]
    ort_only = "--ort-only" in remaining
    verify_ov_only = "--verify-ov-only" in remaining
    remaining = [a for a in remaining if a not in ("--ort-only", "--verify-ov-only")]

    build_types = ["Release", "RelWithDebInfo", "Debug"]
    if remaining:
        build_types = remaining

    # ── Setup ──
    setup_logging(base_dir)
    logger = logging.getLogger("main")
    logger.info(f"Base directory: {base_dir}")
    logger.info(f"Build types:   {build_types}")
    if ort_only:
        logger.info("Mode: ORT-only (skipping OpenVINO phase)")
    if verify_ov_only:
        logger.info("Mode: verify-ov-only (running OV verification only)")

    # ── Create config and run ──
    config = BuildConfig(
        base_dir=base_dir,
        build_types=build_types,
    )

    # Ensure all downstream scripts and subprocesses use the configured proxy.
    for key, value in config.proxy_env.items():
        os.environ[key] = value
    logger.info(
        "Proxy configured: http_proxy=%s, https_proxy=%s",
        config.http_proxy,
        config.https_proxy,
    )

    orchestrator = Orchestrator(config)
    success = orchestrator.run(ort_only=ort_only, verify_ov_only=verify_ov_only)

    if success:
        logger.info("ALL AGENTS COMPLETED SUCCESSFULLY")
    else:
        logger.error("SOME AGENTS FAILED - check build_report.json")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

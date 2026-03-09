"""
Skill: CMake configure, build, and install operations.
Used by BuildAgent and InstallAgent.
"""

import logging
import sys
from pathlib import Path
from typing import List, Optional

from .terminal import TerminalSkill, CommandResult

logger = logging.getLogger(__name__)


class CompilerSkill:
    """CMake-based build operations."""

    @staticmethod
    def find_cmake() -> Optional[str]:
        result = TerminalSkill.run("where cmake", timeout=30)
        if result.success:
            cmake_path = result.stdout.strip().split("\n")[0].strip()
            logger.info(f"[COMPILER] Found cmake: {cmake_path}")
            return cmake_path
        logger.error("[COMPILER] cmake not found in PATH")
        return None

    @staticmethod
    def find_visual_studio() -> Optional[str]:
        vswhere = (
            r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
        )
        if not Path(vswhere).exists():
            logger.warning("[COMPILER] vswhere.exe not found")
            return None

        result = TerminalSkill.run(
            f'"{vswhere}" -latest -property installationPath', timeout=30
        )
        if result.success:
            vs_path = result.stdout.strip()
            logger.info(f"[COMPILER] Found Visual Studio: {vs_path}")
            return vs_path
        return None

    @staticmethod
    def cmake_configure(
        source_dir: str,
        build_dir: str,
        install_dir: str,
        build_type: str,
        generator: str,
        extra_args: List[str],
        timeout: int = 600,
    ) -> CommandResult:
        cmd = (
            f'cmake -S "{source_dir}" -B "{build_dir}" '
            f'-G "{generator}" '
            f"-DCMAKE_BUILD_TYPE={build_type} "
            f'-DCMAKE_INSTALL_PREFIX="{install_dir}" '
            f'{" ".join(extra_args)}'
        )
        logger.info(f"[COMPILER] CMake configure: {build_type}")
        return TerminalSkill.run(cmd, timeout=timeout)

    @staticmethod
    def cmake_build(
        build_dir: str,
        build_type: str,
        parallel_jobs: int = 8,
        timeout: int = 7200,
    ) -> CommandResult:
        cmd = (
            f'cmake --build "{build_dir}" '
            f"--config {build_type} "
            f"--parallel {parallel_jobs}"
        )
        logger.info(f"[COMPILER] CMake build: {build_type}")
        return TerminalSkill.run(cmd, timeout=timeout)

    @staticmethod
    def cmake_install(
        build_dir: str,
        build_type: str,
        timeout: int = 600,
    ) -> CommandResult:
        cmd = (
            f'cmake --install "{build_dir}" '
            f"--config {build_type}"
        )
        logger.info(f"[COMPILER] CMake install: {build_type}")
        return TerminalSkill.run(cmd, timeout=timeout)

    @staticmethod
    def find_python() -> Optional[str]:
        """Find the full path to the Python executable."""
        # Use the same interpreter that is running the agent — this guarantees
        # the venv Python (with 'packaging' etc.) is passed to CMake.
        py_path = sys.executable
        logger.info(f"[COMPILER] Found python (sys.executable): {py_path}")
        return py_path

    @staticmethod
    def check_prerequisites() -> dict:
        """Check all required build tools are available."""
        checks = {}

        r = TerminalSkill.run("git --version", timeout=30)
        checks["git"] = {"found": r.success, "version": r.stdout.strip() if r.success else None}

        r = TerminalSkill.run("cmake --version", timeout=30)
        checks["cmake"] = {
            "found": r.success,
            "version": r.stdout.split("\n")[0].strip() if r.success else None,
        }

        py_path = CompilerSkill.find_python()
        checks["python"] = {"found": py_path is not None, "path": py_path}

        vs_path = CompilerSkill.find_visual_studio()
        checks["visual_studio"] = {"found": vs_path is not None, "path": vs_path}

        return checks

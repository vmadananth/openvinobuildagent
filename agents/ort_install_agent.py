"""
Agent 7: OrtInstallAgent
Task: Install ONNX Runtime with OVEP (pip wheel or cmake install),
      then generate the ovep_install directory for ORT GenAI.

The ovep_install step does:
  cd build/Windows/<BuildType>
  mkdir ovep_<build_type_lower>
  cmake -DCMAKE_INSTALL_PREFIX=ovep_<type> -DCMAKE_INSTALL_CONFIG_NAME=<type> -P cmake_install.cmake
  Then reorganize:
    move bin/* -> lib/
    move include/onnxruntime/* -> include/
    rmdir include/onnxruntime
    rmdir bin

Skills: terminal, filesystem, compiler
Rules: may create install dirs, validates wheel installed correctly
"""

import os
import sys
from pathlib import Path
from .base_agent import BaseAgent, AgentResult
from config import BuildConfig
from rules.constraints import ORT_INSTALL_AGENT_RULES


class OrtInstallAgent(BaseAgent):

    def __init__(self, config: BuildConfig):
        super().__init__(config, ORT_INSTALL_AGENT_RULES)

    @property
    def name(self) -> str:
        return "OrtInstallAgent"

    def execute(self, build_type: str = "Release", device: str = "CPU", **kwargs) -> AgentResult:
        ort_build = str(self.config.ort_build_dir(build_type, device))
        ort_install = str(self.config.ort_install_dir(build_type, device))

        self.fs.ensure_dir(ort_install)

        # ════════════════════════════════════════════════════════
        #  Strategy 1: Find and install the Python wheel
        # ════════════════════════════════════════════════════════
        wheel_installed = False
        wheel_search_dirs = [
            ort_build,
            str(Path(ort_build) / build_type),
            str(Path(ort_build) / build_type / "dist"),
            str(Path(ort_build) / "dist"),
            str(Path(ort_build) / "Windows" / build_type / "dist"),
        ]

        for search_dir in wheel_search_dirs:
            wheels = self.fs.list_dir(search_dir, "**/*.whl")
            ort_wheels = [w for w in wheels if "onnxruntime" in w.lower()]
            if ort_wheels:
                whl = sorted(ort_wheels)[-1]
                self.logger.info(f"[{self.name}] Found wheel: {whl}")
                r = self.terminal.run(
                    f'"{sys.executable}" -m pip install "{whl}" --force-reinstall',
                    timeout=300,
                )
                if r.success:
                    wheel_installed = True
                    break
                else:
                    self.logger.warning(f"[{self.name}] Wheel install failed: {r.stderr}")

        # ════════════════════════════════════════════════════════
        #  Strategy 2: cmake --install fallback
        # ════════════════════════════════════════════════════════
        if not wheel_installed:
            self.logger.info(f"[{self.name}] No wheel found, trying cmake --install...")
            result = self.compiler.cmake_install(
                build_dir=ort_build,
                build_type=build_type,
            )
            if not result.success:
                ort_source = str(self.config.ort_source_dir)
                pip_result = self.terminal.run(
                    f'"{sys.executable}" -m pip install -e "{ort_source}" --no-build-isolation',
                    timeout=self.config.ort_build_timeout,
                )
                if not pip_result.success:
                    return AgentResult(
                        agent_name=self.name,
                        success=False,
                        build_type=build_type,
                        message=f"All install strategies failed [{device}]",
                        errors=[
                            "No .whl found",
                            f"cmake install: {result.stderr[:500]}",
                            f"pip install: {pip_result.stderr[:500]}",
                        ],
                    )

        # ════════════════════════════════════════════════════════
        #  Generate ovep_install for ORT GenAI
        #  Replicates:
        #    cd build\Windows\<BuildType>
        #    mkdir ovep_<type>
        #    cmake -DCMAKE_INSTALL_PREFIX=ovep_<type>
        #          -DCMAKE_INSTALL_CONFIG_NAME=<type>
        #          -P cmake_install.cmake
        #    cd ovep_<type>\bin  && move * ..\lib\.
        #    cd ..\include\onnxruntime && move * ..\.
        #    rmdir include\onnxruntime
        #    rmdir bin
        # ════════════════════════════════════════════════════════
        ovep_result = self._generate_ovep_install(build_type, ort_build, device)

        # ════════════════════════════════════════════════════════
        #  Validate: check that onnxruntime imports
        # ════════════════════════════════════════════════════════
        verify_cmd = (
            f'"{sys.executable}" -c "'
            'import onnxruntime as ort; '
            'print(ort.__version__); '
            'print(ort.get_available_providers())'
            '"'
        )
        verify_result = self.terminal.run(verify_cmd, timeout=60)

        if not verify_result.success:
            return AgentResult(
                agent_name=self.name,
                success=False,
                build_type=build_type,
                message=f"ORT installs but fails to import [{device}]",
                errors=[verify_result.stderr],
            )

        has_ovep = "OpenVINOExecutionProvider" in verify_result.stdout

        return AgentResult(
            agent_name=self.name,
            success=True,
            build_type=build_type,
            message=f"ORT installed [{device}] (OVEP={'YES' if has_ovep else 'NO'}), ovep_install={'OK' if ovep_result else 'SKIPPED'}",
            details={
                "install_dir": ort_install,
                "wheel_installed": wheel_installed,
                "has_ovep": has_ovep,
                "ovep_install_generated": ovep_result,
                "ovep_install_dir": str(self.config.ovep_install_dir(build_type, device)),
                "openvino_device": device,
                "ort_output": verify_result.stdout.strip(),
            },
        )

    def _generate_ovep_install(self, build_type: str, ort_build: str, device: str) -> bool:
        """Generate the OVEP install directory for ORT GenAI integration.

        Equivalent to:
          cd build\\Windows\\<BuildType>
          mkdir ovep_<type>
          cmake -DCMAKE_INSTALL_PREFIX=ovep_<type> -DCMAKE_INSTALL_CONFIG_NAME=<type> -P cmake_install.cmake
          Then reorganize directory structure.
        """
        ovep_name = f"ovep_{build_type.lower()}_{device.lower()}"

        # Find the Windows build subdirectory (build.bat puts output here)
        windows_build_dir = str(Path(ort_build) / "Windows" / build_type)
        if not self.fs.dir_exists(windows_build_dir):
            # Fallback: try directly in ort_build/<build_type>
            windows_build_dir = str(Path(ort_build) / build_type)
        if not self.fs.dir_exists(windows_build_dir):
            # Fallback: use ort_build directly
            windows_build_dir = ort_build

        cmake_install_script = str(Path(windows_build_dir) / "cmake_install.cmake")
        if not self.fs.file_exists(cmake_install_script):
            self.logger.warning(
                f"[{self.name}] cmake_install.cmake not found at {windows_build_dir}, "
                f"skipping ovep_install generation"
            )
            return False

        ovep_dir = str(Path(windows_build_dir) / ovep_name)
        self.fs.ensure_dir(ovep_dir)

        # Step 1: cmake install to ovep_<type>
        self.logger.info(f"[{self.name}] Generating {ovep_name} install...")
        cmake_cmd = (
            f'cmake '
            f'-DCMAKE_INSTALL_PREFIX="{ovep_dir}" '
            f'-DCMAKE_INSTALL_CONFIG_NAME={build_type} '
            f'-P "{cmake_install_script}"'
        )
        result = self.terminal.run(cmake_cmd, cwd=windows_build_dir, timeout=300)
        if not result.success:
            self.logger.error(f"[{self.name}] ovep cmake install failed: {result.stderr[:500]}")
            return False

        # Step 2: Reorganize — move bin/* to lib/
        bin_dir = str(Path(ovep_dir) / "bin")
        lib_dir = str(Path(ovep_dir) / "lib")
        if self.fs.dir_exists(bin_dir):
            self.fs.ensure_dir(lib_dir)
            move_cmd = f'robocopy "{bin_dir}" "{lib_dir}" /MOV /E'
            self.terminal.run(move_cmd, timeout=60)
            # robocopy returns non-zero on success, so just remove bin after
            self.fs.remove_dir(bin_dir)
            self.logger.info(f"[{self.name}] Moved bin/* -> lib/")

        # Step 3: Reorganize — move include/onnxruntime/* to include/
        include_dir = str(Path(ovep_dir) / "include")
        ort_include_dir = str(Path(include_dir) / "onnxruntime")
        if self.fs.dir_exists(ort_include_dir):
            move_cmd = f'robocopy "{ort_include_dir}" "{include_dir}" /MOV /E'
            self.terminal.run(move_cmd, timeout=60)
            self.fs.remove_dir(ort_include_dir)
            self.logger.info(f"[{self.name}] Moved include/onnxruntime/* -> include/")

        self.logger.info(f"[{self.name}] OVEP install generated at: {ovep_dir}")
        return True

"""
Agent 6: OrtBuildAgent
Task: Build ONNX Runtime with OpenVINO Execution Provider (OVEP).
Skills: terminal, filesystem, compiler
Rules: may create build dirs, must set OpenVINO_DIR before building

Uses ORT's build.bat on Windows with these flags:
  build.bat --config <type> --cmake_generator "Visual Studio 18 2026"
    --use_openvino --build_wheel --build_shared_lib --parallel --skip_tests

The OV install from the previous pipeline stage is referenced via OpenVINO_DIR.
"""

import os
from pathlib import Path
from .base_agent import BaseAgent, AgentResult
from config import BuildConfig
from rules.constraints import ORT_BUILD_AGENT_RULES


class OrtBuildAgent(BaseAgent):

    def __init__(self, config: BuildConfig):
        super().__init__(config, ORT_BUILD_AGENT_RULES)

    @property
    def name(self) -> str:
        return "OrtBuildAgent"

    def execute(self, build_type: str = "Release", device: str = "CPU", **kwargs) -> AgentResult:
        ort_source = str(self.config.ort_source_dir)
        ort_build = str(self.config.ort_build_dir(build_type, device))
        ov_install = str(self.config.install_dir_for[build_type])
        ov_flag = self.config.ort_openvino_flag_for(device)

        # Step 1: Verify OpenVINO is installed (prerequisite)
        ov_cmake_dir = str(Path(ov_install) / "runtime" / "cmake")
        if not self.fs.dir_exists(ov_cmake_dir):
            # Fallback: check directly in install dir
            ov_cmake_dir = ov_install
            self.logger.warning(
                f"[{self.name}] runtime/cmake not found, using {ov_cmake_dir}"
            )

        # Step 2: Check build tools
        prereqs = self.compiler.check_prerequisites()
        missing = [k for k, v in prereqs.items() if not v["found"]]
        if missing:
            return AgentResult(
                agent_name=self.name,
                success=False,
                build_type=build_type,
                message=f"Missing prerequisites: {', '.join(missing)}",
                details=prereqs,
            )

        # Step 3: Ensure build directory
        self.fs.ensure_dir(ort_build)

        # Step 4: Build using build.bat (Windows) or build.py (fallback)
        # Exact command:
        #   build.bat --config Release --cmake_generator "Visual Studio 18 2026"
        #     --use_openvino --build_wheel --build_shared_lib --parallel --skip_tests
        #     --build_dir <ort_build> --cmake_extra_defines OpenVINO_DIR=<ov_cmake_dir>
        build_bat = str(Path(ort_source) / "build.bat")
        build_py = str(Path(ort_source) / "tools" / "ci_build" / "build.py")

        if self.fs.file_exists(build_bat):
            # Use build.bat (Windows — recommended)
            cmd = (
                f'"{build_bat}" '
                f'--config {build_type} '
                f'--cmake_generator "{self.config.cmake_generator}" '
                f'--use_openvino '
                f'--build_wheel '
                f'--build_shared_lib '
                f'--parallel '
                f'--skip_tests '
                f'--build_dir "{ort_build}" '
                f'--cmake_extra_defines OpenVINO_DIR="{ov_cmake_dir}"'
            )
        elif self.fs.file_exists(build_py):
            # Fallback: build.py
            cmd = (
                f'python "{build_py}" '
                f'--build_dir "{ort_build}" '
                f'--config {build_type} '
                f'--cmake_generator "{self.config.cmake_generator}" '
                f'--use_openvino '
                f'--build_wheel '
                f'--build_shared_lib '
                f'--parallel '
                f'--skip_tests '
                f'--cmake_extra_defines OpenVINO_DIR="{ov_cmake_dir}"'
            )
        else:
            # Last resort: raw CMake
            ort_cmake_src = str(Path(ort_source) / "cmake")
            ort_install = str(self.config.ort_install_dir(build_type, device))
            cmd = (
                f'cmake -S "{ort_cmake_src}" -B "{ort_build}" '
                f'-G "{self.config.cmake_generator}" '
                f'-DCMAKE_BUILD_TYPE={build_type} '
                f'-DCMAKE_INSTALL_PREFIX="{ort_install}" '
                f'-Donnxruntime_USE_OPENVINO=ON '
                f'-Donnxruntime_BUILD_SHARED_LIB=ON '
                f'-Donnxruntime_BUILD_UNIT_TESTS=OFF'
            )

        # Set OpenVINO_DIR in environment for the build process
        env = os.environ.copy()
        env["OpenVINO_DIR"] = ov_cmake_dir
        env["OPENVINO_DIR"] = ov_install

        self.logger.info(f"[{self.name}] Building ORT-OVEP [{build_type}]...")
        self.logger.info(f"[{self.name}] OpenVINO_DIR = {ov_cmake_dir}")

        build_result = self.terminal.run(
            cmd,
            timeout=self.config.ort_build_timeout,
            env=env,
        )

        if not build_result.success:
            return AgentResult(
                agent_name=self.name,
                success=False,
                build_type=build_type,
                message=f"ORT build failed [{device}] ({build_result.duration_seconds:.0f}s)",
                errors=[build_result.stderr],
                details=build_result.to_dict(),
            )

        # If we used raw CMake (last resort), also run the build step
        used_raw_cmake = not self.fs.file_exists(build_bat) and not self.fs.file_exists(build_py)
        if used_raw_cmake:
            cmake_build_result = self.terminal.run(
                f'cmake --build "{ort_build}" --config {build_type} '
                f'--parallel {self.config.parallel_jobs}',
                timeout=self.config.ort_build_timeout,
                env=env,
            )
            if not cmake_build_result.success:
                return AgentResult(
                    agent_name=self.name,
                    success=False,
                    build_type=build_type,
                    message=f"ORT cmake build failed [{device}] ({cmake_build_result.duration_seconds:.0f}s)",
                    errors=[cmake_build_result.stderr],
                    details=cmake_build_result.to_dict(),
                )

        return AgentResult(
            agent_name=self.name,
            success=True,
            build_type=build_type,
            message=f"ORT-OVEP build succeeded [{device}] in {build_result.duration_seconds:.0f}s",
            details={
                "build_dir": ort_build,
                "openvino_dir": ov_cmake_dir,
                "openvino_device": device,
                "duration": build_result.duration_seconds,
            },
        )

"""
Agent 2: BuildAgent
Task: CMake configure + build for a given build type.
Skills: terminal, filesystem, compiler
Rules: may create build dirs, must check prerequisites first
"""

from .base_agent import BaseAgent, AgentResult
from config import BuildConfig
from rules.constraints import BUILD_AGENT_RULES


class BuildAgent(BaseAgent):

    def __init__(self, config: BuildConfig):
        super().__init__(config, BUILD_AGENT_RULES)

    @property
    def name(self) -> str:
        return "BuildAgent"

    def execute(self, build_type: str = "Release", **kwargs) -> AgentResult:
        source_dir = str(self.config.source_dir)
        build_dir = str(self.config.build_dir_for[build_type])
        install_dir = str(self.config.install_dir_for[build_type])

        # Step 1: Check prerequisites
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

        # Step 2: Resolve Python executable path
        python_exe = self.config.python_executable
        if not python_exe:
            python_exe = self.compiler.find_python()
        if not python_exe:
            return AgentResult(
                agent_name=self.name,
                success=False,
                build_type=build_type,
                message="Cannot find Python executable",
            )

        # Build cmake args with Python3_EXECUTABLE
        cmake_args = list(self.config.cmake_extra_args)
        cmake_args.append(f'-DPython3_EXECUTABLE="{python_exe}"')

        # OpenVINO Python bindings (pyopenvino) intentionally cannot be built
        # in Debug mode — disable them to avoid a hard MSBuild error.
        if build_type == "Debug":
            cmake_args = [a for a in cmake_args if "-DENABLE_PYTHON" not in a and "-DENABLE_WHEEL" not in a]
            cmake_args += ["-DENABLE_PYTHON=OFF", "-DENABLE_WHEEL=OFF"]

        # Step 3: Create build directory
        self.fs.ensure_dir(build_dir)

        # Step 4: CMake configure
        # cmake . -Bbuild -G "VS 17 2022" -DCMAKE_BUILD_TYPE=<type>
        #   -DENABLE_DEBUG_CAPS=ON -DENABLE_PYTHON=ON
        #   -DPython3_EXECUTABLE="<path>" -DENABLE_WHEEL=ON -DENABLE_TESTS=ON
        self.logger.info(f"[{self.name}] Configuring {build_type}...")
        self.logger.info(f"[{self.name}] Python3_EXECUTABLE = {python_exe}")
        config_result = self.compiler.cmake_configure(
            source_dir=source_dir,
            build_dir=build_dir,
            install_dir=install_dir,
            build_type=build_type,
            generator=self.config.cmake_generator,
            extra_args=cmake_args,
        )

        if not config_result.success:
            return AgentResult(
                agent_name=self.name,
                success=False,
                build_type=build_type,
                message="CMake configure failed",
                errors=[config_result.stderr],
                details=config_result.to_dict(),
            )

        # Step 4: CMake build
        self.logger.info(f"[{self.name}] Building {build_type}...")
        build_result = self.compiler.cmake_build(
            build_dir=build_dir,
            build_type=build_type,
            parallel_jobs=self.config.parallel_jobs,
            timeout=self.config.build_timeout,
        )

        if not build_result.success:
            # MSBuild writes errors to stdout on Windows; fall back if stderr is empty
            error_output = build_result.stderr.strip() or build_result.stdout.strip()
            return AgentResult(
                agent_name=self.name,
                success=False,
                build_type=build_type,
                message=f"Build failed ({build_result.duration_seconds:.0f}s)",
                errors=[error_output],
                details=build_result.to_dict(),
            )

        return AgentResult(
            agent_name=self.name,
            success=True,
            build_type=build_type,
            message=f"Build succeeded in {build_result.duration_seconds:.0f}s",
            details={"build_dir": build_dir, "duration": build_result.duration_seconds},
        )

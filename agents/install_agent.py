"""
Agent 3: InstallAgent
Task: Run cmake --install for a given build type.
Skills: terminal, filesystem, compiler
Rules: may create install dirs, validates key files exist after install
"""

from pathlib import Path
from .base_agent import BaseAgent, AgentResult
from config import BuildConfig
from rules.constraints import INSTALL_AGENT_RULES


class InstallAgent(BaseAgent):

    def __init__(self, config: BuildConfig):
        super().__init__(config, INSTALL_AGENT_RULES)

    @property
    def name(self) -> str:
        return "InstallAgent"

    def execute(self, build_type: str = "Release", **kwargs) -> AgentResult:
        build_dir = str(self.config.build_dir_for[build_type])
        install_dir = str(self.config.install_dir_for[build_type])

        self.fs.ensure_dir(install_dir)

        # Run cmake --install
        self.logger.info(f"[{self.name}] Installing {build_type}...")
        result = self.compiler.cmake_install(
            build_dir=build_dir,
            build_type=build_type,
        )

        if not result.success:
            return AgentResult(
                agent_name=self.name,
                success=False,
                build_type=build_type,
                message="cmake --install failed",
                errors=[result.stderr],
                details=result.to_dict(),
            )

        # Validate: check that install dir has real content
        all_files = self.fs.list_dir(install_dir, "**/*")
        if len(all_files) < 5:
            return AgentResult(
                agent_name=self.name,
                success=False,
                build_type=build_type,
                message=f"Install dir has very few files ({len(all_files)})",
                errors=["Installation may be incomplete"],
            )

        # Install Python wheel if present
        wheel_dir = Path(install_dir) / "tools"
        if self.fs.dir_exists(str(wheel_dir)):
            wheels = self.fs.list_dir(str(wheel_dir), "*.whl")
            for whl in wheels:
                self.terminal.run(f'pip install "{whl}" --force-reinstall', timeout=120)

        return AgentResult(
            agent_name=self.name,
            success=True,
            build_type=build_type,
            message=f"Installed to {install_dir}",
            details={"install_dir": install_dir, "file_count": len(all_files)},
        )

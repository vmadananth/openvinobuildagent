"""
Agent 5: OrtDownloadAgent
Task: Clone the ONNX Runtime repository with all submodules.
Skills: terminal, filesystem
Rules: read-only (no file edits), must validate build.bat exists
"""

from pathlib import Path
from .base_agent import BaseAgent, AgentResult
from config import BuildConfig
from rules.constraints import ORT_DOWNLOAD_AGENT_RULES


class OrtDownloadAgent(BaseAgent):

    def __init__(self, config: BuildConfig):
        super().__init__(config, ORT_DOWNLOAD_AGENT_RULES)

    @property
    def name(self) -> str:
        return "OrtDownloadAgent"

    def execute(self, **kwargs) -> AgentResult:
        source_dir = str(self.config.ort_source_dir)

        # Check disk space
        free_gb = self.fs.get_disk_space_gb(self.config.base_dir)
        if free_gb < 30:
            return AgentResult(
                agent_name=self.name,
                success=False,
                message=f"Insufficient disk space: {free_gb:.1f} GB free, need 30+ GB",
            )

        self.fs.ensure_dir(self.config.base_dir)

        # If already cloned, just pull latest
        if self.fs.dir_exists(str(Path(source_dir) / ".git")):
            self.logger.info(f"[{self.name}] ORT repo already exists, pulling latest...")
            result = self.terminal.run(
                f'git -C "{source_dir}" pull --rebase',
                timeout=self.config.clone_timeout,
            )
            if not result.success:
                return AgentResult(
                    agent_name=self.name,
                    success=False,
                    message="Git pull failed",
                    errors=[result.stderr],
                )
        else:
            # Fresh clone with submodules
            cmd = (
                f'git clone --recursive '
                f'--branch {self.config.ort_branch} '
                f'"{self.config.ort_repo_url}" "{source_dir}"'
            )
            result = self.terminal.run(cmd, timeout=self.config.clone_timeout)
            if not result.success:
                return AgentResult(
                    agent_name=self.name,
                    success=False,
                    message="Git clone failed",
                    errors=[result.stderr],
                )

        # Init submodules
        sub_result = self.terminal.run(
            f'git -C "{source_dir}" submodule update --init --recursive',
            timeout=self.config.clone_timeout,
        )

        # Validate: build.bat must exist at repo root
        build_bat = str(Path(source_dir) / "build.bat")
        if not self.fs.file_exists(build_bat):
            return AgentResult(
                agent_name=self.name,
                success=False,
                message="Clone incomplete: build.bat missing",
            )

        return AgentResult(
            agent_name=self.name,
            success=True,
            message=f"ONNX Runtime cloned to {source_dir}",
            details={
                "source_dir": source_dir,
                "submodules_updated": sub_result.success,
            },
        )

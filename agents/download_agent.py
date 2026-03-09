"""
Agent 1: DownloadAgent
Task: Clone the OpenVINO repo with all submodules.
Skills: terminal, filesystem
Rules: read-only (no file edits), must validate CMakeLists.txt exists
"""

from pathlib import Path
from .base_agent import BaseAgent, AgentResult
from config import BuildConfig
from rules.constraints import DOWNLOAD_AGENT_RULES


class DownloadAgent(BaseAgent):

    def __init__(self, config: BuildConfig):
        super().__init__(config, DOWNLOAD_AGENT_RULES)

    @property
    def name(self) -> str:
        return "DownloadAgent"

    def execute(self, **kwargs) -> AgentResult:
        source_dir = str(self.config.source_dir)

        # Rule: check disk space first
        free_gb = self.fs.get_disk_space_gb(self.config.base_dir)
        if free_gb < 20:
            return AgentResult(
                agent_name=self.name,
                success=False,
                message=f"Insufficient disk space: {free_gb:.1f} GB free, need 20+ GB",
            )

        # Ensure base directory exists
        self.fs.ensure_dir(self.config.base_dir)

        # If already cloned, just pull latest
        if self.fs.dir_exists(str(Path(source_dir) / ".git")):
            self.logger.info(f"[{self.name}] Repo already exists, pulling latest...")
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
            # Fresh clone
            cmd = (
                f'git clone --recurse-submodules --shallow-submodules '
                f'--branch {self.config.repo_branch} '
                f'"{self.config.repo_url}" "{source_dir}"'
            )
            result = self.terminal.run(cmd, timeout=self.config.clone_timeout)
            if not result.success:
                return AgentResult(
                    agent_name=self.name,
                    success=False,
                    message="Git clone failed",
                    errors=[result.stderr],
                )

        # Init submodules explicitly
        sub_result = self.terminal.run(
            f'git -C "{source_dir}" submodule update --init --recursive',
            timeout=self.config.clone_timeout,
        )

        # Validate: CMakeLists.txt must exist
        cmake_file = str(Path(source_dir) / "CMakeLists.txt")
        if not self.fs.file_exists(cmake_file):
            return AgentResult(
                agent_name=self.name,
                success=False,
                message="Clone incomplete: CMakeLists.txt missing",
            )

        return AgentResult(
            agent_name=self.name,
            success=True,
            message=f"Repository cloned to {source_dir}",
            details={
                "source_dir": source_dir,
                "submodules_updated": sub_result.success,
            },
        )

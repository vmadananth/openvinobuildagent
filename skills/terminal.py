"""
Skill: Execute shell commands on the host machine.
Used by agents that need to run git, cmake, python, etc.
"""

import subprocess
import logging
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CommandResult:
    """Structured result from a terminal command."""
    command: str
    return_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    success: bool

    def to_dict(self) -> dict:
        return {
            "command": self.command,
            "return_code": self.return_code,
            "stdout": self.stdout[-5000:],
            "stderr": self.stderr[-5000:],
            "duration_seconds": round(self.duration_seconds, 2),
            "success": self.success,
        }


class TerminalSkill:
    """Run shell commands and return structured results."""

    @staticmethod
    def run(
        command: str,
        cwd: Optional[str] = None,
        timeout: int = 3600,
        env: Optional[dict] = None,
        shell: bool = True,
    ) -> CommandResult:
        logger.info(f"[TERMINAL] Running: {command}")
        if cwd:
            logger.info(f"[TERMINAL] CWD: {cwd}")

        start = time.time()
        try:
            proc = subprocess.run(
                command,
                cwd=cwd,
                shell=shell,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
            duration = time.time() - start
            result = CommandResult(
                command=command,
                return_code=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
                duration_seconds=duration,
                success=(proc.returncode == 0),
            )
        except subprocess.TimeoutExpired:
            duration = time.time() - start
            result = CommandResult(
                command=command,
                return_code=-1,
                stdout="",
                stderr=f"TIMEOUT after {timeout}s",
                duration_seconds=duration,
                success=False,
            )
        except Exception as e:
            duration = time.time() - start
            result = CommandResult(
                command=command,
                return_code=-1,
                stdout="",
                stderr=str(e),
                duration_seconds=duration,
                success=False,
            )

        level = logging.INFO if result.success else logging.ERROR
        logger.log(level, f"[TERMINAL] Exit code: {result.return_code} ({result.duration_seconds:.1f}s)")
        if not result.success:
            logger.error(f"[TERMINAL] STDERR: {result.stderr[-1000:]}")

        return result

    @staticmethod
    def run_multiple(commands: list, cwd: Optional[str] = None,
                     stop_on_failure: bool = True) -> list:
        """Run a sequence of commands, optionally stopping on first failure."""
        results = []
        for cmd in commands:
            r = TerminalSkill.run(cmd, cwd=cwd)
            results.append(r)
            if not r.success and stop_on_failure:
                logger.error(f"[TERMINAL] Stopping sequence at: {cmd}")
                break
        return results

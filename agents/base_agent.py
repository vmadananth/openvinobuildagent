"""
Base agent class. All agents inherit from this.
Provides: skill bindings, retry logic, structured results.
"""

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from rules.constraints import AgentRules
from skills import TerminalSkill, FilesystemSkill, CompilerSkill
from config import BuildConfig


@dataclass
class AgentResult:
    """Standardized result every agent must return."""
    agent_name: str
    success: bool
    build_type: Optional[str] = None
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    errors: list = field(default_factory=list)

    def __str__(self):
        status = "PASS" if self.success else "FAIL"
        variant = f" [{self.build_type}]" if self.build_type else ""
        return f"[{status}] {self.agent_name}{variant}: {self.message}"


class BaseAgent(ABC):
    """
    Base class for all agents.
    
    Each agent has:
      - skills: tools it can use (terminal, filesystem, compiler)
      - rules:  constraints on its behavior (from AgentRules)
    
    The run() method wraps execute() with retry logic.
    """

    def __init__(self, config: BuildConfig, rules: AgentRules):
        self.config = config
        self.rules = rules
        self.logger = logging.getLogger(self.__class__.__name__)

        # Keep proxy env available for all command executions.
        for key, value in self.config.proxy_env.items():
            os.environ[key] = value

        # Bind skills (agents access tools through these)
        self.terminal = TerminalSkill()
        self.fs = FilesystemSkill()
        self.compiler = CompilerSkill()

    @property
    @abstractmethod
    def name(self) -> str:
        """Human readable agent name."""
        ...

    @abstractmethod
    def execute(self, **kwargs) -> AgentResult:
        """Run this agent's primary task. Subclasses implement this."""
        ...

    def run(self, **kwargs) -> AgentResult:
        """Run the agent with retry logic from rules."""
        self.logger.info(f"{'='*60}")
        self.logger.info(f"[{self.name}] Starting (rules: {self.rules.description})")
        self.logger.info(f"{'='*60}")

        last_result = None
        for attempt in range(1, self.rules.max_retries + 1):
            self.logger.info(f"[{self.name}] Attempt {attempt}/{self.rules.max_retries}")
            try:
                result = self.execute(**kwargs)
                last_result = result

                if result.success:
                    self.logger.info(f"[{self.name}] {result}")
                    return result
                else:
                    self.logger.warning(f"[{self.name}] Attempt {attempt} failed: {result.message}")
            except Exception as e:
                self.logger.exception(f"[{self.name}] Exception on attempt {attempt}")
                last_result = AgentResult(
                    agent_name=self.name,
                    success=False,
                    message=f"Exception: {e}",
                    errors=[str(e)],
                )

        self.logger.error(f"[{self.name}] All {self.rules.max_retries} attempts failed")
        return last_result

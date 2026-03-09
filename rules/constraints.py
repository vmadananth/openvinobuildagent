"""
Rules: Constraints that define what each agent is allowed to do.
Each agent gets a rule set that limits its skills and behavior.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class AgentRules:
    """Rules that constrain agent behavior."""

    # What skills this agent is allowed to use
    allowed_skills: List[str] = field(default_factory=list)

    # Permissions
    can_write_files: bool = False
    can_run_commands: bool = False
    can_delete: bool = False

    # Retry policy
    max_retries: int = 2

    # Must validate output before reporting success?
    must_validate: bool = True

    # Human-readable description
    description: str = ""

    def check_skill_allowed(self, skill_name: str) -> bool:
        if not self.allowed_skills:
            return True
        return skill_name in self.allowed_skills

    def __str__(self) -> str:
        return (
            f"Rules({self.description}): "
            f"skills={self.allowed_skills}, "
            f"write={self.can_write_files}, "
            f"commands={self.can_run_commands}, "
            f"retries={self.max_retries}"
        )


# ── Pre-defined rule sets for each agent ──

DOWNLOAD_AGENT_RULES = AgentRules(
    allowed_skills=["terminal", "filesystem"],
    can_write_files=False,
    can_run_commands=True,
    can_delete=False,
    max_retries=2,
    must_validate=True,
    description="Clone repo only; no file modifications",
)

BUILD_AGENT_RULES = AgentRules(
    allowed_skills=["terminal", "filesystem", "compiler"],
    can_write_files=True,
    can_run_commands=True,
    can_delete=False,
    max_retries=1,
    must_validate=True,
    description="Configure and build; may create build directories",
)

INSTALL_AGENT_RULES = AgentRules(
    allowed_skills=["terminal", "filesystem", "compiler"],
    can_write_files=True,
    can_run_commands=True,
    can_delete=False,
    max_retries=1,
    must_validate=True,
    description="Install built artifacts",
)

ORT_DOWNLOAD_AGENT_RULES = AgentRules(
    allowed_skills=["terminal", "filesystem"],
    can_write_files=False,
    can_run_commands=True,
    can_delete=False,
    max_retries=2,
    must_validate=True,
    description="Clone ONNX Runtime repo only; no file modifications",
)

ORT_BUILD_AGENT_RULES = AgentRules(
    allowed_skills=["terminal", "filesystem", "compiler"],
    can_write_files=True,
    can_run_commands=True,
    can_delete=False,
    max_retries=1,
    must_validate=True,
    description="Build ORT with OVEP; requires OpenVINO installed first",
)

ORT_INSTALL_AGENT_RULES = AgentRules(
    allowed_skills=["terminal", "filesystem", "compiler"],
    can_write_files=True,
    can_run_commands=True,
    can_delete=False,
    max_retries=2,
    must_validate=True,
    description="Install ORT wheel or cmake artifacts",
)

VERIFY_AGENT_RULES = AgentRules(
    allowed_skills=["terminal", "filesystem"],
    can_write_files=True,   # writes test scripts
    can_run_commands=True,
    can_delete=False,
    max_retries=2,
    must_validate=True,
    description="Write and run verification tests",
)

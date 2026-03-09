"""
Skill: File and directory operations.
Used by agents that need to create dirs, read/write files, check disk space.
"""

import shutil
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class FilesystemSkill:
    """File and directory operations."""

    @staticmethod
    def ensure_dir(path: str) -> bool:
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            logger.info(f"[FS] Ensured directory: {path}")
            return True
        except Exception as e:
            logger.error(f"[FS] Failed to create {path}: {e}")
            return False

    @staticmethod
    def dir_exists(path: str) -> bool:
        return Path(path).is_dir()

    @staticmethod
    def file_exists(path: str) -> bool:
        return Path(path).is_file()

    @staticmethod
    def remove_dir(path: str) -> bool:
        try:
            if Path(path).exists():
                shutil.rmtree(path)
                logger.info(f"[FS] Removed directory: {path}")
            return True
        except Exception as e:
            logger.error(f"[FS] Failed to remove {path}: {e}")
            return False

    @staticmethod
    def write_file(path: str, content: str) -> bool:
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info(f"[FS] Wrote file: {path}")
            return True
        except Exception as e:
            logger.error(f"[FS] Failed to write {path}: {e}")
            return False

    @staticmethod
    def read_file(path: str) -> Optional[str]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.error(f"[FS] Failed to read {path}: {e}")
            return None

    @staticmethod
    def list_dir(path: str, pattern: str = "*") -> list:
        try:
            return [str(p) for p in Path(path).glob(pattern)]
        except Exception as e:
            logger.error(f"[FS] Failed to list {path}: {e}")
            return []

    @staticmethod
    def get_disk_space_gb(path: str) -> float:
        try:
            target = Path(path).resolve()
            # disk_usage needs an existing path; fall back to the target drive root.
            usage_path = target if target.exists() else Path(target.anchor or Path.cwd().anchor)
            usage = shutil.disk_usage(usage_path)
            return usage.free / (1024 ** 3)
        except Exception:
            return 0.0

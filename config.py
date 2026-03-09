"""
Configuration for the OpenVINO multi-agent build system.
All paths are derived from a single user-provided base_dir.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict


@dataclass
class BuildConfig:
    """Configuration for the entire build pipeline."""

    # User-provided input: base directory for everything
    base_dir: str

    # Repository
    repo_url: str = "https://github.com/openvinotoolkit/openvino.git"
    repo_branch: str = "master"

    # Build variants to produce
    build_types: List[str] = field(default_factory=lambda: [
        "Release", "RelWithDebInfo", "Debug"
    ])

    # CMake options for OpenVINO
    cmake_generator: str = "Visual Studio 18 2026"
    cmake_extra_args: List[str] = field(default_factory=lambda: [
        "-DENABLE_DEBUG_CAPS=ON",
        "-DENABLE_PYTHON=ON",
        "-DENABLE_WHEEL=ON",
        "-DENABLE_TESTS=ON",
        "-DENABLE_INTEL_GPU=ON",
    ])

    # Python executable path (auto-detected if empty)
    python_executable: str = ""

    # Network proxy settings.
    # The system reads http_proxy / https_proxy (or their upper-case variants)
    # from the environment. Set those variables before running main.py if your
    # network requires a proxy (e.g. export http_proxy=http://proxy.example.com:911).
    # Leave them unset for direct / no-proxy environments.
    http_proxy: str = field(default_factory=lambda: (
        (os.getenv("http_proxy") or os.getenv("HTTP_PROXY") or "")
        .replace("http:://", "http://")
    ))
    https_proxy: str = field(default_factory=lambda: (
        (os.getenv("https_proxy") or os.getenv("HTTPS_PROXY") or "")
        .replace("https:://", "https://")
        .replace("http:://", "http://")
    ))

    # Parallelism
    parallel_jobs: int = 8

    # Timeouts (seconds)
    clone_timeout: int = 1200
    build_timeout: int = 7200
    test_timeout: int = 300

    # ── ONNX Runtime + OVEP settings ──
    ort_repo_url: str = "https://github.com/intel/onnxruntime.git"
    ort_branch: str = "ovep-develop"
    # Build ORT+OVEP variants for each listed OpenVINO device.
    ort_openvino_devices: List[str] = field(default_factory=lambda: ["CPU", "GPU"])
    # Verify native OV and OVEP on these devices.
    openvino_verify_devices: List[str] = field(default_factory=lambda: ["CPU", "GPU"])
    ort_build_timeout: int = 7200
    # Build ORT only for these config types.
    ort_build_types: List[str] = field(default_factory=lambda: ["Release", "RelWithDebInfo"])

    # Derived paths — OpenVINO
    @property
    def source_dir(self) -> Path:
        return Path(self.base_dir) / "openvino_source"

    @property
    def build_dir_for(self) -> dict:
        return {
            bt: Path(self.base_dir) / "build" / "openvino" / bt
            for bt in self.build_types
        }

    @property
    def install_dir_for(self) -> dict:
        return {
            bt: Path(self.base_dir) / "install" / "openvino" / bt
            for bt in self.build_types
        }

    # Derived paths — ONNX Runtime
    @property
    def ort_source_dir(self) -> Path:
        return Path(self.base_dir) / "onnxruntime_source"

    @property
    def ort_build_dir_for(self) -> dict:
        return {
            bt: self.ort_build_dir(bt)
            for bt in self.build_types
        }

    @property
    def ort_install_dir_for(self) -> dict:
        return {
            bt: self.ort_install_dir(bt)
            for bt in self.build_types
        }

    # Derived paths — OVEP install (reorganized for ORT GenAI)
    @property
    def ovep_install_dir_for(self) -> dict:
        """Maps build_type to OVEP install prefix name used by cmake_install.
        e.g. Release -> ovep_release, Debug -> ovep_debug"""
        return {
            bt: self.ovep_install_dir(bt)
            for bt in self.build_types
        }

    def ort_build_dir(self, build_type: str, device: str = "CPU") -> Path:
        return Path(self.base_dir) / "build" / "onnxruntime" / build_type / device.upper()

    def ort_install_dir(self, build_type: str, device: str = "CPU") -> Path:
        return Path(self.base_dir) / "install" / "onnxruntime" / build_type / device.upper()

    def ovep_install_dir(self, build_type: str, device: str = "CPU") -> Path:
        return self.ort_build_dir(build_type, device) / "Windows" / build_type / f"ovep_{build_type.lower()}_{device.lower()}"

    @staticmethod
    def ort_openvino_flag_for(device: str) -> str:
        device_upper = device.upper()
        if device_upper == "CPU":
            return "CPU_FP32"
        if device_upper == "GPU":
            return "GPU_FP32"
        return device_upper

    @property
    def proxy_env(self) -> Dict[str, str]:
        """Proxy environment variables for tools that honor lower/upper case names."""
        return {
            "http_proxy": self.http_proxy,
            "https_proxy": self.https_proxy,
            "HTTP_PROXY": self.http_proxy,
            "HTTPS_PROXY": self.https_proxy,
        }

"""
OV Suite — Test 1: Import and device enumeration

Verifies that the openvino package is importable, Core can be instantiated,
and the requested device is reported as available.

Applies to: Release, RelWithDebInfo
Skipped for: Debug (no Python bindings)
"""

import pytest


def test_openvino_importable():
    """openvino package must be importable without error."""
    from openvino import Core  # noqa: F401


def test_core_instantiation():
    """Core() must construct without raising an exception."""
    from openvino import Core
    core = Core()
    assert core is not None


def test_available_devices():
    """Core.available_devices must return at least one device."""
    from openvino import Core
    devices = Core().available_devices
    assert devices, f"Expected at least one device, got: {devices}"
    print(f"Available devices: {devices}")


def test_requested_device_present(ov_device):
    """The device specified by OV_DEVICE must appear in available_devices."""
    from openvino import Core
    devices = Core().available_devices
    assert ov_device in devices, (
        f"Device '{ov_device}' not available. Available: {devices}"
    )


def test_openvino_version():
    """openvino.__version__ must be a non-empty string."""
    import openvino
    assert openvino.__version__, "Version string is empty"
    print(f"OpenVINO version: {openvino.__version__}")

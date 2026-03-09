"""
OVEP Suite — Test 1: Provider presence and session creation

Verifies that onnxruntime was built with OpenVINOExecutionProvider and that
an InferenceSession can be created using it.

Applies to: Release, RelWithDebInfo
Skipped for: Debug (no OV Python bindings → OVEP unavailable)
"""

import pytest


def test_openvinoexecutionprovider_in_list():
    """OpenVINOExecutionProvider must appear in ort.get_available_providers()."""
    import onnxruntime as ort

    providers = ort.get_available_providers()
    print(f"Available providers: {providers}")
    assert "OpenVINOExecutionProvider" in providers, (
        f"OpenVINOExecutionProvider not found. Available: {providers}\n"
        "Ensure the onnxruntime_openvino wheel was installed."
    )


def test_ort_version_accessible():
    """ORT __version__ must be a non-empty string."""
    import onnxruntime as ort
    assert ort.__version__
    print(f"ORT version: {ort.__version__}")


def test_session_creates_with_ovep(ort_device_flag, relu_onnx_path):
    """InferenceSession with OpenVINOExecutionProvider must construct without error."""
    import onnxruntime as ort

    sess = ort.InferenceSession(
        relu_onnx_path,
        providers=[("OpenVINOExecutionProvider", {"device_type": ort_device_flag})],
    )
    assert sess is not None


def test_ovep_input_names_accessible(ort_device_flag, relu_onnx_path):
    """Session input names must be accessible after creation with OVEP."""
    import onnxruntime as ort

    sess = ort.InferenceSession(
        relu_onnx_path,
        providers=[("OpenVINOExecutionProvider", {"device_type": ort_device_flag})],
    )
    names = [i.name for i in sess.get_inputs()]
    assert "X" in names, f"Expected input 'X', got: {names}"

"""
ORT Suite — Test 1: Import, version, and provider enumeration

Verifies that onnxruntime is importable, CPUExecutionProvider is present,
and a basic InferenceSession can be constructed.

Applies to: Release, RelWithDebInfo, Debug
"""


def test_onnxruntime_importable():
    """onnxruntime must be importable without error."""
    import onnxruntime  # noqa: F401


def test_version_accessible():
    """onnxruntime.__version__ must be a non-empty string."""
    import onnxruntime as ort
    assert ort.__version__, "ORT version string is empty"
    print(f"ORT version: {ort.__version__}")


def test_cpu_provider_present():
    """CPUExecutionProvider must always be in the available providers list."""
    import onnxruntime as ort
    providers = ort.get_available_providers()
    assert "CPUExecutionProvider" in providers, (
        f"CPUExecutionProvider missing. Available: {providers}"
    )


def test_providers_list_non_empty():
    """get_available_providers() must return a non-empty list."""
    import onnxruntime as ort
    providers = ort.get_available_providers()
    assert isinstance(providers, list) and providers, \
        "Provider list must be a non-empty list"
    print(f"Available providers: {providers}")


def test_inference_session_constructable(relu_onnx_path):
    """InferenceSession must be constructable with CPUExecutionProvider."""
    import onnxruntime as ort
    sess = ort.InferenceSession(relu_onnx_path, providers=["CPUExecutionProvider"])
    assert sess is not None

"""
OVEP Suite — Test 3: Numerical parity — OVEP vs CPUExecutionProvider

Results from OpenVINOExecutionProvider must agree with CPUExecutionProvider
within tolerance (OVEP may use lower internal precision on some ops).

Tolerance: atol=1e-4 (conservative to account for FP16 path on GPU)

Applies to: Release, RelWithDebInfo
Skipped for: Debug
"""

import numpy as np
import pytest

_ATOL = 1e-4


def _cpu_session(onnx_path: str):
    import onnxruntime as ort
    return ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])


def _ovep_session(onnx_path: str, device_flag: str):
    import onnxruntime as ort
    return ort.InferenceSession(
        onnx_path,
        providers=[("OpenVINOExecutionProvider", {"device_type": device_flag})],
    )


# ── per-model parity tests ────────────────────────────────────────────────────

def test_relu_ovep_matches_cpu(ort_device_flag, relu_onnx_path):
    """ReLU: OVEP and CPU results must be identical within tolerance."""
    x      = np.array([[-3.0, 0.0, 2.5]], dtype=np.float32)
    cpu_r  = _cpu_session(relu_onnx_path).run(None, {"X": x})[0]
    ovep_r = _ovep_session(relu_onnx_path, ort_device_flag).run(None, {"X": x})[0]
    np.testing.assert_allclose(ovep_r, cpu_r, atol=_ATOL,
                               err_msg="ReLU: OVEP vs CPU mismatch")


def test_softmax_ovep_matches_cpu(ort_device_flag, softmax_onnx_path):
    """Softmax: OVEP and CPU results must agree within tolerance."""
    x      = np.array([[2.0, 1.0, 0.5, 3.0, 1.5]], dtype=np.float32)
    cpu_r  = _cpu_session(softmax_onnx_path).run(None, {"X": x})[0]
    ovep_r = _ovep_session(softmax_onnx_path, ort_device_flag).run(None, {"X": x})[0]
    np.testing.assert_allclose(ovep_r, cpu_r, atol=_ATOL,
                               err_msg="Softmax: OVEP vs CPU mismatch")


def test_matmul_bias_ovep_matches_cpu(ort_device_flag, matmul_bias_onnx_path):
    """MatMul+Bias: OVEP and CPU results must agree."""
    x      = np.array([[0.5, -1.0, 2.0, 0.3]], dtype=np.float32)
    cpu_r  = _cpu_session(matmul_bias_onnx_path).run(None, {"X": x})[0]
    ovep_r = _ovep_session(matmul_bias_onnx_path, ort_device_flag).run(None, {"X": x})[0]
    np.testing.assert_allclose(ovep_r, cpu_r, atol=_ATOL,
                               err_msg="MatMul+Bias: OVEP vs CPU mismatch")


def test_random_batch_ovep_matches_cpu(ort_device_flag, relu_onnx_path):
    """Five random inputs: OVEP and CPU must agree on every call."""
    cpu_sess  = _cpu_session(relu_onnx_path)
    ovep_sess = _ovep_session(relu_onnx_path, ort_device_flag)
    rng       = np.random.default_rng(42)
    for i in range(5):
        x = rng.standard_normal((1, 3)).astype(np.float32)
        np.testing.assert_allclose(
            ovep_sess.run(None, {"X": x})[0],
            cpu_sess.run(None, {"X": x})[0],
            atol=_ATOL,
            err_msg=f"Random input #{i}: OVEP vs CPU mismatch",
        )


def test_softmax_sum_preserved_ovep(ort_device_flag, softmax_onnx_path):
    """OVEP softmax output must sum to 1 regardless of input scale."""
    import onnxruntime as ort

    sess = _ovep_session(softmax_onnx_path, ort_device_flag)
    for scale in [0.01, 1.0, 100.0]:
        x      = np.array([[scale, scale * 2, scale * 0.5, scale * 3, scale]], dtype=np.float32)
        result = sess.run(None, {"X": x})[0]
        np.testing.assert_allclose(result.sum(), 1.0, atol=1e-5,
                                   err_msg=f"Softmax sum != 1 at scale {scale}")

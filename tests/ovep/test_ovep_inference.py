"""
OVEP Suite — Test 2: Inference correctness via OpenVINOExecutionProvider

Runs ReLU, MatMul+Bias, and Softmax models through ORT using OVEP and
validates numerical output against numpy reference values.

Applies to: Release, RelWithDebInfo
Skipped for: Debug
"""

import numpy as np
import pytest


def test_relu_ovep(ort_device_flag, relu_onnx_path):
    """ReLU via OVEP: negative values must be clipped to zero."""
    import onnxruntime as ort

    sess   = ort.InferenceSession(
        relu_onnx_path,
        providers=[("OpenVINOExecutionProvider", {"device_type": ort_device_flag})],
    )
    x      = np.array([[-2.0, 0.0, 3.0]], dtype=np.float32)
    result = sess.run(None, {"X": x})[0]
    np.testing.assert_allclose(result, np.maximum(x, 0), atol=1e-5,
                               err_msg="ReLU OVEP output mismatch")


def test_matmul_bias_ovep(ort_device_flag, matmul_bias_onnx_path):
    """Linear layer via OVEP: Y = X @ W + B must match numpy reference."""
    import onnxruntime as ort

    w        = np.array([[0.5, -0.3], [0.1, 0.8], [-0.2, 0.4], [0.6, -0.1]], dtype=np.float32)
    b        = np.array([0.1, -0.2], dtype=np.float32)
    sess     = ort.InferenceSession(
        matmul_bias_onnx_path,
        providers=[("OpenVINOExecutionProvider", {"device_type": ort_device_flag})],
    )
    x        = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    result   = sess.run(None, {"X": x})[0]
    np.testing.assert_allclose(result, x @ w + b, atol=1e-4,
                               err_msg="MatMul+Bias OVEP output mismatch")


def test_softmax_ovep(ort_device_flag, softmax_onnx_path):
    """Softmax via OVEP: outputs must be positive and sum to 1."""
    import onnxruntime as ort

    sess   = ort.InferenceSession(
        softmax_onnx_path,
        providers=[("OpenVINOExecutionProvider", {"device_type": ort_device_flag})],
    )
    x      = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
    result = sess.run(None, {"X": x})[0]
    assert np.all(result > 0), "All softmax outputs must be positive"
    np.testing.assert_allclose(result.sum(), 1.0, atol=1e-5,
                               err_msg="Softmax OVEP outputs must sum to 1")


def test_relu_random_inputs_ovep(ort_device_flag, relu_onnx_path):
    """Five random inputs via OVEP must satisfy max(x, 0) within tolerance."""
    import onnxruntime as ort

    sess = ort.InferenceSession(
        relu_onnx_path,
        providers=[("OpenVINOExecutionProvider", {"device_type": ort_device_flag})],
    )
    rng  = np.random.default_rng(99)
    for _ in range(5):
        x      = rng.standard_normal((1, 3)).astype(np.float32)
        result = sess.run(None, {"X": x})[0]
        np.testing.assert_allclose(result, np.maximum(x, 0), atol=1e-5)

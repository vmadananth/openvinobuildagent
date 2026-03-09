"""
ORT Suite — Test 2: Inference correctness on CPUExecutionProvider

Runs ReLU, MatMul+Bias, and Softmax models through ORT and validates the
output numerically against numpy reference values.

Applies to: Release, RelWithDebInfo, Debug
"""

import numpy as np
import pytest


def test_relu_cpu(relu_onnx_path):
    """ReLU: negative values must be clipped to zero."""
    import onnxruntime as ort

    sess   = ort.InferenceSession(relu_onnx_path, providers=["CPUExecutionProvider"])
    x      = np.array([[-2.0, 0.0, 3.0]], dtype=np.float32)
    result = sess.run(None, {"X": x})[0]
    np.testing.assert_allclose(result, np.maximum(x, 0), atol=1e-7,
                               err_msg="ReLU output mismatch")


def test_matmul_bias_cpu(matmul_bias_onnx_path):
    """Linear layer: Y = X @ W + B must match numpy reference."""
    import onnxruntime as ort

    w        = np.array([[0.5, -0.3], [0.1, 0.8], [-0.2, 0.4], [0.6, -0.1]], dtype=np.float32)
    b        = np.array([0.1, -0.2], dtype=np.float32)
    sess     = ort.InferenceSession(matmul_bias_onnx_path, providers=["CPUExecutionProvider"])
    x        = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    result   = sess.run(None, {"X": x})[0]
    np.testing.assert_allclose(result, x @ w + b, atol=1e-5,
                               err_msg="MatMul+Bias output mismatch")


def test_softmax_cpu(softmax_onnx_path):
    """Softmax: all outputs must be positive and sum to 1."""
    import onnxruntime as ort

    sess   = ort.InferenceSession(softmax_onnx_path, providers=["CPUExecutionProvider"])
    x      = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
    result = sess.run(None, {"X": x})[0]
    assert np.all(result > 0), "All softmax outputs must be positive"
    np.testing.assert_allclose(result.sum(), 1.0, atol=1e-6,
                               err_msg="Softmax outputs must sum to 1")


def test_relu_zero_boundary(relu_onnx_path):
    """ReLU applied to an all-zeros input must return all zeros."""
    import onnxruntime as ort

    sess   = ort.InferenceSession(relu_onnx_path, providers=["CPUExecutionProvider"])
    x      = np.zeros((1, 3), dtype=np.float32)
    result = sess.run(None, {"X": x})[0]
    np.testing.assert_array_equal(result, np.zeros_like(x))


def test_relu_negative_only(relu_onnx_path):
    """ReLU on all-negative input must produce all zeros."""
    import onnxruntime as ort

    sess   = ort.InferenceSession(relu_onnx_path, providers=["CPUExecutionProvider"])
    x      = np.array([[-1.0, -2.0, -0.001]], dtype=np.float32)
    result = sess.run(None, {"X": x})[0]
    np.testing.assert_array_equal(result, np.zeros_like(x))


def test_repeated_inferences_consistent(relu_onnx_path):
    """Repeated calls with the same input must yield identical results."""
    import onnxruntime as ort

    sess = ort.InferenceSession(relu_onnx_path, providers=["CPUExecutionProvider"])
    x    = np.array([[1.0, -1.0, 0.0]], dtype=np.float32)
    r1   = sess.run(None, {"X": x})[0]
    r2   = sess.run(None, {"X": x})[0]
    np.testing.assert_array_equal(r1, r2, err_msg="Non-deterministic output detected")


def test_random_inputs_cpu(relu_onnx_path):
    """Five independently seeded random inputs must all satisfy max(x, 0)."""
    import onnxruntime as ort

    sess = ort.InferenceSession(relu_onnx_path, providers=["CPUExecutionProvider"])
    rng  = np.random.default_rng(7)
    for _ in range(5):
        x      = rng.standard_normal((1, 3)).astype(np.float32)
        result = sess.run(None, {"X": x})[0]
        np.testing.assert_allclose(result, np.maximum(x, 0), atol=1e-7)

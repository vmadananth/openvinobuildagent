"""
OV Suite — Test 2: Inference correctness

Compiles three ONNX models via openvino.Core.compile_model() and validates
numerical output against numpy reference values.

Models tested:
  - ReLU         : Y = max(X, 0)
  - MatMul+Bias  : Y = X @ W + B  (linear layer)
  - Softmax      : Y = softmax(X, axis=1)

Applies to: Release, RelWithDebInfo
Skipped for: Debug (no Python bindings)
"""

import numpy as np
import pytest


def test_relu_inference(ov_device, relu_onnx_path):
    """ReLU: negative values must be clipped to 0, positives unchanged."""
    from openvino import Core

    compiled = Core().compile_model(relu_onnx_path, ov_device)
    x        = np.array([[-2.0, 0.0, 3.0]], dtype=np.float32)
    result   = compiled([x])[0]
    np.testing.assert_allclose(result, np.maximum(x, 0), atol=1e-6,
                               err_msg="ReLU output mismatch")


def test_matmul_bias_inference(ov_device, matmul_bias_onnx_path):
    """Linear layer: result must match the numpy reference X @ W + B."""
    from openvino import Core

    w        = np.array([[0.5, -0.3], [0.1, 0.8], [-0.2, 0.4], [0.6, -0.1]], dtype=np.float32)
    b        = np.array([0.1, -0.2], dtype=np.float32)
    compiled = Core().compile_model(matmul_bias_onnx_path, ov_device)
    x        = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    result   = compiled([x])[0]
    np.testing.assert_allclose(result, x @ w + b, atol=1e-5,
                               err_msg="MatMul+Bias output mismatch")


def test_softmax_inference(ov_device, softmax_onnx_path):
    """Softmax: all outputs must be positive and sum to 1."""
    from openvino import Core

    compiled = Core().compile_model(softmax_onnx_path, ov_device)
    x        = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
    result   = compiled([x])[0]
    assert np.all(result > 0), "All softmax outputs must be positive"
    np.testing.assert_allclose(result.sum(), 1.0, atol=1e-6,
                               err_msg="Softmax outputs must sum to 1")


def test_output_dtype_float32(ov_device, relu_onnx_path):
    """Compiled model output must preserve the float32 precision of the input."""
    from openvino import Core

    compiled = Core().compile_model(relu_onnx_path, ov_device)
    x        = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    result   = compiled([x])[0]
    assert result.dtype == np.float32, f"Expected float32 output, got {result.dtype}"


def test_inference_determinism(ov_device, relu_onnx_path):
    """Running the same input twice must produce identical results."""
    from openvino import Core

    compiled = Core().compile_model(relu_onnx_path, ov_device)
    x  = np.array([[0.5, -1.0, 2.0]], dtype=np.float32)
    r1 = compiled([x])[0]
    r2 = compiled([x])[0]
    np.testing.assert_array_equal(r1, r2, err_msg="Non-deterministic output detected")

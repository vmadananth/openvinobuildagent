"""
OV Suite — Test 3: Dynamic shapes and multi-batch inference

Verifies that OpenVINO can compile a model with a dynamic (None) batch
dimension and produce correct results for both single-sample and multi-sample
batches.

Applies to: Release, RelWithDebInfo
Skipped for: Debug (no Python bindings)
"""

import numpy as np
import pytest
from pathlib import Path


@pytest.fixture(scope="module")
def relu_dynamic_onnx_path(tmp_path_factory) -> str:
    """ReLU model with a dynamic batch dimension: shape [None, 4]."""
    import onnx
    from onnx import helper, TensorProto

    X     = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, 4])
    Y     = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, 4])
    node  = helper.make_node("Relu", ["X"], ["Y"])
    graph = helper.make_graph([node], "relu_dyn", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    path  = tmp_path_factory.mktemp("dyn") / "relu_dynamic.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture(scope="module")
def multi_op_onnx_path(tmp_path_factory) -> str:
    """Three-op pipeline: ReLU -> Add(bias) -> Abs.  Input/output shape [1, 4]."""
    import onnx
    from onnx import helper, TensorProto, numpy_helper

    bias  = np.array([-0.5, 0.5, -1.0, 1.0], dtype=np.float32)
    X     = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    Y     = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])
    B     = numpy_helper.from_array(bias, "Bias")
    relu  = helper.make_node("Relu", ["X"], ["R"])
    add   = helper.make_node("Add",  ["R", "Bias"], ["A"])
    abs_  = helper.make_node("Abs",  ["A"], ["Y"])
    graph = helper.make_graph([relu, add, abs_], "multi_op", [X], [Y], initializer=[B])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    path  = tmp_path_factory.mktemp("dyn") / "multi_op.onnx"
    onnx.save(model, str(path))
    return str(path)


def test_dynamic_batch_single(ov_device, relu_dynamic_onnx_path):
    """Dynamic-batch model compiled and run with batch size 1."""
    from openvino import Core

    compiled = Core().compile_model(relu_dynamic_onnx_path, ov_device)
    x        = np.array([[-1.0, 0.5, 2.0, -3.0]], dtype=np.float32)
    result   = compiled([x])[0]
    np.testing.assert_allclose(result, np.maximum(x, 0), atol=1e-6)


def test_dynamic_batch_multi(ov_device, relu_dynamic_onnx_path):
    """Same dynamic-batch model run with batch size 4 (runtime reshape)."""
    from openvino import Core

    compiled = Core().compile_model(relu_dynamic_onnx_path, ov_device)
    rng      = np.random.default_rng(0)
    x        = rng.standard_normal((4, 4)).astype(np.float32)
    result   = compiled([x])[0]
    np.testing.assert_allclose(result, np.maximum(x, 0), atol=1e-6)


def test_multi_op_pipeline(ov_device, multi_op_onnx_path):
    """Three-op pipeline: output must equal abs(relu(X) + bias)."""
    from openvino import Core

    bias     = np.array([-0.5, 0.5, -1.0, 1.0], dtype=np.float32)
    compiled = Core().compile_model(multi_op_onnx_path, ov_device)
    x        = np.array([[1.0, -1.0, 2.0, -2.0]], dtype=np.float32)
    result   = compiled([x])[0]
    expected = np.abs(np.maximum(x, 0) + bias)
    np.testing.assert_allclose(result, expected, atol=1e-6,
                               err_msg="Multi-op pipeline output mismatch")


def test_random_inputs(ov_device, relu_onnx_path):
    """Five random inputs must all match the numpy reference."""
    from openvino import Core

    compiled = Core().compile_model(relu_onnx_path, ov_device)
    rng      = np.random.default_rng(42)
    for _ in range(5):
        x        = rng.standard_normal((1, 3)).astype(np.float32)
        result   = compiled([x])[0]
        np.testing.assert_allclose(result, np.maximum(x, 0), atol=1e-6)

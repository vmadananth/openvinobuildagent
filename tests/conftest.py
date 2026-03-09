"""
Shared pytest fixtures for the OV / ORT / OVEP inference test suites.

Environment variables (configured automatically by run_tests.py):
  OV_INSTALL_DIR   Path to the OpenVINO install tree for the tested build type
                   e.g.  C:\\ov_build\\install\\openvino\\Release
  BUILD_TYPE       Release | RelWithDebInfo | Debug          (default: Release)
  OV_DEVICE        CPU | GPU                                 (default: CPU)
  ORT_DEVICE       CPU | GPU                                 (default: CPU)
"""

import os
import numpy as np
import pytest
from pathlib import Path


# ── environment fixtures ──────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def build_type() -> str:
    return os.environ.get("BUILD_TYPE", "Release")


@pytest.fixture(scope="session")
def ov_install_dir(build_type) -> Path:
    """Path to the OV install tree.  Tests that depend on this skip if absent."""
    val = os.environ.get("OV_INSTALL_DIR", "")
    if not val:
        pytest.skip(
            "OV_INSTALL_DIR is not set — run via run_tests.py "
            "or set the variable manually before invoking pytest."
        )
    p = Path(val)
    if not p.is_dir():
        pytest.fail(f"OV_INSTALL_DIR points to a missing directory: {p}")
    return p


@pytest.fixture(scope="session")
def ov_device() -> str:
    return os.environ.get("OV_DEVICE", "CPU").upper()


@pytest.fixture(scope="session")
def ort_device_flag() -> str:
    """OVEP device type string accepted by ORT (e.g. CPU_FP32, GPU_FP32)."""
    raw = os.environ.get("ORT_DEVICE", "CPU").upper()
    return {"CPU": "CPU_FP32", "GPU": "GPU_FP32"}.get(raw, raw)


# ── shared ONNX model builders ────────────────────────────────────────────────

@pytest.fixture(scope="session")
def relu_onnx_path(tmp_path_factory) -> str:
    """Single-op ReLU model: Y = max(X, 0), shape [1, 3]."""
    import onnx
    from onnx import helper, TensorProto

    X     = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])
    Y     = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])
    node  = helper.make_node("Relu", ["X"], ["Y"])
    graph = helper.make_graph([node], "relu", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    path  = tmp_path_factory.mktemp("models") / "relu.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture(scope="session")
def matmul_bias_onnx_path(tmp_path_factory) -> str:
    """Linear layer: Y = X @ W + B, input shape [1, 4] -> output [1, 2]."""
    import onnx
    from onnx import helper, TensorProto, numpy_helper

    w = np.array([[0.5, -0.3], [0.1, 0.8], [-0.2, 0.4], [0.6, -0.1]], dtype=np.float32)
    b = np.array([0.1, -0.2], dtype=np.float32)

    X      = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    Y      = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])
    W      = numpy_helper.from_array(w, "W")
    B      = numpy_helper.from_array(b, "B")
    matmul = helper.make_node("MatMul", ["X", "W"], ["XW"])
    add    = helper.make_node("Add", ["XW", "B"], ["Y"])
    graph  = helper.make_graph([matmul, add], "linear", [X], [Y], initializer=[W, B])
    model  = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    path   = tmp_path_factory.mktemp("models") / "matmul_bias.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture(scope="session")
def softmax_onnx_path(tmp_path_factory) -> str:
    """Softmax model: Y = softmax(X), shape [1, 5]."""
    import onnx
    from onnx import helper, TensorProto

    X     = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 5])
    Y     = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 5])
    node  = helper.make_node("Softmax", ["X"], ["Y"], axis=1)
    graph = helper.make_graph([node], "softmax", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    path  = tmp_path_factory.mktemp("models") / "softmax.onnx"
    onnx.save(model, str(path))
    return str(path)

"""
Agent 4: VerifyAgent
Task: Run verification tests against installed OpenVINO and/or ORT-OVEP.
  phase="openvino":
    Test 1: Native OpenVINO Python API
    Test 2: OpenVINO loading an ONNX model
    Test 3: Basic ONNX Runtime inference (CPU fallback)
  phase="ovep":
    Test 4: ORT with OpenVINOExecutionProvider (must be present)
    Test 5: OVEP correctness on a real ONNX model
    Test 6: OVEP vs CPU provider comparison

Skills: terminal, filesystem (writes test scripts, then runs them)
Rules: may write test files, must report pass/fail per test
"""

import sys
from pathlib import Path
from .base_agent import BaseAgent, AgentResult
from config import BuildConfig
from rules.constraints import VERIFY_AGENT_RULES


class VerifyAgent(BaseAgent):

    def __init__(self, config: BuildConfig):
        super().__init__(config, VERIFY_AGENT_RULES)

    @property
    def name(self) -> str:
        return "VerifyAgent"

    def execute(self, build_type: str = "Release", phase: str = "openvino", **kwargs) -> AgentResult:
        device = kwargs.get("device", "CPU")
        if phase == "ovep":
            test_dir = str(Path(self.config.base_dir) / "verification_tests" / phase / build_type / device)
        else:
            test_dir = str(Path(self.config.base_dir) / "verification_tests" / phase / build_type)
        self.fs.ensure_dir(test_dir)

        if phase == "openvino":
            return self._verify_openvino(build_type, test_dir)
        elif phase == "ovep":
            return self._verify_ovep(build_type, test_dir, device)
        else:
            return AgentResult(
                agent_name=self.name,
                success=False,
                build_type=build_type,
                message=f"Unknown phase: {phase}",
            )

    # ══════════════════════════════════════════════════════════════════
    #  PHASE 1: OpenVINO verification tests
    # ══════════════════════════════════════════════════════════════════

    def _verify_openvino(self, build_type: str, test_dir: str) -> AgentResult:
        import os
        results = {}
        all_passed = True

        # Build a test environment mirroring what setupvars.bat sets
        ov_install = Path(self.config.install_dir_for[build_type])
        test_env = os.environ.copy()

        # PYTHONPATH: <install>/python and <install>/python/python3
        ov_python_dir = str(ov_install / "python")
        ov_python3_dir = str(ov_install / "python" / "python3")
        existing_py = test_env.get("PYTHONPATH", "")
        test_env["PYTHONPATH"] = os.pathsep.join(
            p for p in [ov_python_dir, ov_python3_dir, existing_py] if p
        )

        # OPENVINO_LIB_PATHS: runtime bins + TBB
        # The bin subfolder matches the build type (Release, RelWithDebInfo, Debug).
        # Fall back to Release/Debug if the exact subfolder doesn't exist.
        intel64_dir = ov_install / "runtime" / "bin" / "intel64"
        bin_candidates = [build_type, "Release", "Debug"]
        lib_paths = []
        seen = set()
        for candidate in bin_candidates:
            p = intel64_dir / candidate
            if str(p) not in seen:
                lib_paths.append(str(p))
                seen.add(str(p))
        for tbb_candidate in [
            ov_install / "runtime" / "3rdparty" / "tbb" / "redist" / "intel64" / "vc14",
            ov_install / "runtime" / "3rdparty" / "tbb" / "bin" / "intel64" / "vc14",
            ov_install / "runtime" / "3rdparty" / "tbb" / "bin",
        ]:
            if tbb_candidate.exists():
                lib_paths.append(str(tbb_candidate))
                break
        test_env["OPENVINO_LIB_PATHS"] = os.pathsep.join(lib_paths)
        test_env["PATH"] = os.pathsep.join(lib_paths) + os.pathsep + test_env.get("PATH", "")

        # ── Test 1: Native OpenVINO Python API ──
        test1_code = '''\
import sys
try:
    from openvino import Core
    core = Core()
    devices = core.available_devices
    print(f"OpenVINO loaded successfully")
    print(f"Available devices: {devices}")
    assert len(devices) > 0, "No devices found"
    print("TEST_NATIVE_OV: PASSED")
except Exception as e:
    print(f"TEST_NATIVE_OV: FAILED - {e}")
    sys.exit(1)
'''
        test1_file = str(Path(test_dir) / "test_native_ov.py")
        self.fs.write_file(test1_file, test1_code)
        r1 = self.terminal.run(f'"{sys.executable}" "{test1_file}"', timeout=self.config.test_timeout, env=test_env)
        results["native_ov"] = {
            "passed": r1.success and "PASSED" in r1.stdout,
            "output": r1.stdout,
            "error": r1.stderr,
        }
        if not results["native_ov"]["passed"]:
            all_passed = False

        # ── Test 2: OpenVINO + ONNX Frontend ──
        test2_code = f'''\
import sys
import numpy as np
try:
    from openvino import Core
    import onnx
    from onnx import helper, TensorProto

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])
    node = helper.make_node("Relu", inputs=["X"], outputs=["Y"])
    graph = helper.make_graph([node], "test_relu", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx_path = "test_relu.onnx"
    onnx.save(model, onnx_path)

    core = Core()
    available_devices = set(core.available_devices)
    target_devices = {self.config.openvino_verify_devices!r}
    input_data = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)
    expected = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)

    for device in target_devices:
        assert device in available_devices, (
            f"Required OpenVINO device '{{device}}' not available. "
            f"Available: {{sorted(available_devices)}}"
        )
        compiled = core.compile_model(onnx_path, device)
        result = compiled([input_data])[0]
        assert np.allclose(result, expected), f"Wrong result on {{device}}: {{result}}"
        print(f"{{device}}: PASSED")
    print("TEST_OV_ONNX: PASSED")
except Exception as e:
    print(f"TEST_OV_ONNX: FAILED - {{e}}")
    sys.exit(1)
'''
        test2_file = str(Path(test_dir) / "test_ov_onnx.py")
        self.fs.write_file(test2_file, test2_code)
        r2 = self.terminal.run(f'"{sys.executable}" "{test2_file}"', cwd=test_dir, timeout=self.config.test_timeout, env=test_env)
        results["ov_onnx"] = {
            "passed": r2.success and "PASSED" in r2.stdout,
            "output": r2.stdout,
            "error": r2.stderr,
        }
        if not results["ov_onnx"]["passed"]:
            all_passed = False

        # ── Test 3: Basic ONNX Runtime inference ──
        test3_code = '''\
import sys
import numpy as np
try:
    import onnxruntime as ort
    import onnx
    from onnx import helper, TensorProto
except ModuleNotFoundError:
    print("TEST_ORT_CPU: SKIPPED - onnxruntime not installed yet")
    sys.exit(0)

try:

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])
    node = helper.make_node("Relu", inputs=["X"], outputs=["Y"])
    graph = helper.make_graph([node], "test_relu", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx_path = "test_relu_ort.onnx"
    onnx.save(model, onnx_path)

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_data = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)
    result = session.run(None, {"X": input_data})[0]
    expected = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    assert np.allclose(result, expected), f"Wrong result: {result}"
    print("TEST_ORT_CPU: PASSED")
except Exception as e:
    print(f"TEST_ORT_CPU: FAILED - {e}")
    sys.exit(1)
'''
        test3_file = str(Path(test_dir) / "test_ort_cpu.py")
        self.fs.write_file(test3_file, test3_code)
        r3 = self.terminal.run(f'"{sys.executable}" "{test3_file}"', cwd=test_dir, timeout=self.config.test_timeout, env=test_env)
        results["ort_cpu"] = {
            "passed": r3.success and ("PASSED" in r3.stdout or "SKIPPED" in r3.stdout),
            "output": r3.stdout,
            "error": r3.stderr,
        }
        if not results["ort_cpu"]["passed"]:
            all_passed = False

        passed = sum(1 for v in results.values() if v["passed"])
        return AgentResult(
            agent_name=self.name,
            success=all_passed,
            build_type=build_type,
            message=f"OV Verification: {passed}/{len(results)} tests passed",
            details=results,
            errors=[f"{k}: {v['error']}" for k, v in results.items() if not v["passed"]],
        )

    # ══════════════════════════════════════════════════════════════════
    #  PHASE 2: ONNX Runtime + OpenVINO EP verification tests
    # ══════════════════════════════════════════════════════════════════

    def _verify_ovep(self, build_type: str, test_dir: str, device: str) -> AgentResult:
        import os
        results = {}
        all_passed = True

        # Build a test environment mirroring what setupvars.bat sets
        ov_install = Path(self.config.install_dir_for[build_type])
        test_env = os.environ.copy()

        # PYTHONPATH: <install>/python and <install>/python/python3
        ov_python_dir = str(ov_install / "python")
        ov_python3_dir = str(ov_install / "python" / "python3")
        existing_py = test_env.get("PYTHONPATH", "")
        test_env["PYTHONPATH"] = os.pathsep.join(
            p for p in [ov_python_dir, ov_python3_dir, existing_py] if p
        )

        # OPENVINO_LIB_PATHS: runtime bins
        intel64_dir = ov_install / "runtime" / "bin" / "intel64"
        bin_candidates = [build_type, "Release", "Debug"]
        lib_paths = []
        seen = set()
        for candidate in bin_candidates:
            p = intel64_dir / candidate
            if str(p) not in seen:
                lib_paths.append(str(p))
                seen.add(str(p))
        for tbb_candidate in [
            ov_install / "runtime" / "3rdparty" / "tbb" / "redist" / "intel64" / "vc14",
            ov_install / "runtime" / "3rdparty" / "tbb" / "bin" / "intel64" / "vc14",
            ov_install / "runtime" / "3rdparty" / "tbb" / "bin",
        ]:
            if tbb_candidate.exists():
                lib_paths.append(str(tbb_candidate))
                break
        test_env["OPENVINO_LIB_PATHS"] = os.pathsep.join(lib_paths)
        test_env["PATH"] = os.pathsep.join(lib_paths) + os.pathsep + test_env.get("PATH", "")

        # ── Test 4: OVEP must be present ──
        test4_code = '''\
import sys
try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    print(f"ORT version: {ort.__version__}")
    print(f"Available providers: {providers}")
    assert "OpenVINOExecutionProvider" in providers, (
        f"OpenVINOExecutionProvider NOT found! Available: {providers}"
    )
    print("TEST_OVEP_PRESENT: PASSED")
except Exception as e:
    print(f"TEST_OVEP_PRESENT: FAILED - {e}")
    sys.exit(1)
'''
        test4_file = str(Path(test_dir) / "test_ovep_present.py")
        self.fs.write_file(test4_file, test4_code)
        r4 = self.terminal.run(f'"{sys.executable}" "{test4_file}"', timeout=self.config.test_timeout, env=test_env)
        results["ovep_present"] = {
            "passed": r4.success and "PASSED" in r4.stdout,
            "output": r4.stdout,
            "error": r4.stderr,
        }
        if not results["ovep_present"]["passed"]:
            all_passed = False

        # ── Test 5: OVEP inference correctness ──
        ort_device = self.config.ort_openvino_flag_for(device)

        test5_code = f'''\
import sys
import numpy as np
try:
    import onnxruntime as ort
    import onnx
    from onnx import helper, TensorProto, numpy_helper

    # Build a small model: Y = MatMul(X, W) + B (linear layer)
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])

    w_data = np.array([[0.5, -0.3], [0.1, 0.8], [-0.2, 0.4], [0.6, -0.1]], dtype=np.float32)
    b_data = np.array([0.1, -0.2], dtype=np.float32)

    W = numpy_helper.from_array(w_data, name="W")
    B = numpy_helper.from_array(b_data, name="B")

    matmul = helper.make_node("MatMul", ["X", "W"], ["XW"])
    add = helper.make_node("Add", ["XW", "B"], ["Y"])

    graph = helper.make_graph(
        [matmul, add], "linear_model", [X], [Y], initializer=[W, B]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx_path = "test_linear_ovep.onnx"
    onnx.save(model, onnx_path)

    # Run on OVEP
    session = ort.InferenceSession(
        onnx_path,
        providers=[("OpenVINOExecutionProvider", {{"device_type": "{ort_device}"}})],
    )
    input_data = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    result = session.run(None, {{"X": input_data}})[0]

    # Expected: X @ W + B
    expected = input_data @ w_data + b_data
    assert np.allclose(result, expected, atol=1e-5), (
        f"Mismatch!\\nGot:      {{result}}\\nExpected: {{expected}}"
    )
    print(f"Result: {{result}}")
    print("TEST_OVEP_CORRECTNESS_{device.upper()}: PASSED")
except Exception as e:
    print(f"TEST_OVEP_CORRECTNESS_{device.upper()}: FAILED - {{e}}")
    sys.exit(1)
'''
        test5_file = str(Path(test_dir) / "test_ovep_correctness.py")
        self.fs.write_file(test5_file, test5_code)
        r5 = self.terminal.run(f'"{sys.executable}" "{test5_file}"', cwd=test_dir, timeout=self.config.test_timeout, env=test_env)
        results[f"ovep_correctness_{device.lower()}"] = {
            "passed": r5.success and "PASSED" in r5.stdout,
            "output": r5.stdout,
            "error": r5.stderr,
        }
        if not results[f"ovep_correctness_{device.lower()}"]["passed"]:
            all_passed = False

        # ── Test 6: OVEP vs CPU comparison ──
        test6_code = f'''\
import sys
import numpy as np
try:
    import onnxruntime as ort
    import onnx
    from onnx import helper, TensorProto

    # Simple Softmax model
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 5])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 5])
    node = helper.make_node("Softmax", ["X"], ["Y"], axis=1)
    graph = helper.make_graph([node], "softmax_model", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx_path = "test_softmax.onnx"
    onnx.save(model, onnx_path)

    input_data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)

    # Run on CPU
    cpu_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    cpu_result = cpu_session.run(None, {{"X": input_data}})[0]

    # Run on OVEP
    ovep_session = ort.InferenceSession(
        onnx_path,
        providers=[("OpenVINOExecutionProvider", {{"device_type": "{ort_device}"}})],
    )
    ovep_result = ovep_session.run(None, {{"X": input_data}})[0]

    print(f"CPU result:  {{cpu_result}}")
    print(f"OVEP result: {{ovep_result}}")

    assert np.allclose(cpu_result, ovep_result, atol=1e-5), (
        f"Results differ!\\nCPU:  {{cpu_result}}\\nOVEP: {{ovep_result}}"
    )
    print("TEST_OVEP_VS_CPU_{device.upper()}: PASSED")
except Exception as e:
    print(f"TEST_OVEP_VS_CPU_{device.upper()}: FAILED - {{e}}")
    sys.exit(1)
'''
        test6_file = str(Path(test_dir) / "test_ovep_vs_cpu.py")
        self.fs.write_file(test6_file, test6_code)
        r6 = self.terminal.run(f'"{sys.executable}" "{test6_file}"', cwd=test_dir, timeout=self.config.test_timeout, env=test_env)
        results[f"ovep_vs_cpu_{device.lower()}"] = {
            "passed": r6.success and "PASSED" in r6.stdout,
            "output": r6.stdout,
            "error": r6.stderr,
        }
        if not results[f"ovep_vs_cpu_{device.lower()}"]["passed"]:
            all_passed = False

        passed = sum(1 for v in results.values() if v["passed"])
        return AgentResult(
            agent_name=self.name,
            success=all_passed,
            build_type=build_type,
            message=f"OVEP Verification [{device}]: {passed}/{len(results)} tests passed",
            details=results,
            errors=[f"{k}: {v['error']}" for k, v in results.items() if not v["passed"]],
        )

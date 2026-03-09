# Inference Verification Tests

Standalone pytest-based test suite that verifies inference correctness for:

- **OV** — Native OpenVINO Python API (`openvino.Core`)
- **ORT** — ONNX Runtime with `CPUExecutionProvider`
- **OVEP** — ONNX Runtime with `OpenVINOExecutionProvider`

across **Release**, **RelWithDebInfo**, and **Debug** build configurations.

---

## Directory Layout

```
tests/
├── conftest.py                   Shared fixtures (ONNX model builders, env fixtures)
├── run_tests.py                  CLI test runner — sets up environment and launches pytest
│
├── ov/                           Native OpenVINO tests
│   ├── conftest.py               Injects OV Python paths; skips suite on Debug
│   ├── test_ov_import.py         Import, Core instantiation, device enumeration
│   ├── test_ov_inference.py      ReLU / MatMul+Bias / Softmax correctness
│   └── test_ov_dynamic_shapes.py Dynamic batch dimensions, multi-op pipeline
│
├── ort/                          ORT CPU tests (all build types)
│   ├── test_ort_import.py        Import, version, CPUExecutionProvider presence
│   └── test_ort_inference.py     ReLU / MatMul+Bias / Softmax correctness
│
└── ovep/                         ORT + OpenVINO EP tests
    ├── conftest.py               Injects OV Python paths; skips suite on Debug
    ├── test_ovep_provider.py     OVEP provider presence, session creation
    ├── test_ovep_inference.py    ReLU / MatMul+Bias / Softmax via OVEP
    └── test_ovep_vs_cpu.py       Numerical parity: OVEP results vs CPU results
```

---

## Prerequisites

Python packages required (all already in the project `requirements.txt` plus `pytest`):

```powershell
pip install pytest numpy onnx onnxruntime
```

Or install everything via the project venv (if main.py has already been run once):

```powershell
.venv\Scripts\python.exe -m pip install pytest
```

> **Note:** The `onnxruntime_openvino` wheel (OVEP build) must be installed for the
> OVEP suite. This is done automatically by the build pipeline (`OrtInstallAgent`).

---

## Running Tests

All examples below use `run_tests.py`, which handles environment setup
(`PYTHONPATH`, `PATH`, `OPENVINO_LIB_PATHS`) automatically.

### Full suite — all build types

```powershell
# Release — OV + ORT + OVEP
python tests\run_tests.py --base-dir C:\ov_build --build-type Release

# RelWithDebInfo
python tests\run_tests.py --base-dir C:\ov_build --build-type RelWithDebInfo

# Debug — ORT (CPU) only; OV and OVEP skipped automatically
python tests\run_tests.py --base-dir C:\ov_build --build-type Debug
```

### Run a specific suite

```powershell
# Native OpenVINO only
python tests\run_tests.py --base-dir C:\ov_build --build-type Release --suite ov

# ORT (CPU) only
python tests\run_tests.py --base-dir C:\ov_build --build-type Release --suite ort

# OVEP only
python tests\run_tests.py --base-dir C:\ov_build --build-type Release --suite ovep
```

### GPU device

```powershell
# Run OV and OVEP tests targeting the Intel GPU
python tests\run_tests.py --base-dir C:\ov_build --build-type Release ^
    --ov-device GPU --ort-device GPU --suite ovep
```

### Verbose output

```powershell
python tests\run_tests.py --base-dir C:\ov_build --build-type Release --verbose
```

### Direct pytest invocation (manual env setup)

Set environment variables manually and invoke pytest directly if preferred:

```powershell
$env:OV_INSTALL_DIR = "C:\ov_build\install\openvino\Release"
$env:BUILD_TYPE     = "Release"
$env:OV_DEVICE      = "CPU"
$env:ORT_DEVICE     = "CPU"
$env:PYTHONPATH     = "$env:OV_INSTALL_DIR\python;$env:OV_INSTALL_DIR\python\python3;$env:PYTHONPATH"
$env:PATH           = "$env:OV_INSTALL_DIR\runtime\bin\intel64\Release;$env:PATH"

python -m pytest tests\ -v --tb=short
```

---

## CLI Reference — run_tests.py

| Argument | Default | Description |
|---|---|---|
| `--base-dir` | *(required)* | Root build directory (e.g. `C:\ov_build`) |
| `--build-type` | `Release` | `Release` \| `RelWithDebInfo` \| `Debug` |
| `--suite` | `all` | `ov` \| `ort` \| `ovep` \| `all` |
| `--ov-device` | `CPU` | OpenVINO device for native OV and OVEP tests |
| `--ort-device` | `CPU` | Device string forwarded to ORT OVEP (`CPU_FP32` / `GPU_FP32`) |
| `--verbose` / `-v` | off | Pass `-v` to pytest |

---

## Test Coverage

### OV Suite (`tests/ov/`) — Release & RelWithDebInfo only

| File | Tests |
|---|---|
| `test_ov_import.py` | Import openvino, Core instantiation, device list, version string |
| `test_ov_inference.py` | ReLU correctness, MatMul+Bias vs numpy, Softmax sum=1, float32 dtype, determinism |
| `test_ov_dynamic_shapes.py` | Dynamic batch × 1, dynamic batch × 4, 3-op pipeline (ReLU→Add→Abs), random inputs |

### ORT Suite (`tests/ort/`) — all build types

| File | Tests |
|---|---|
| `test_ort_import.py` | Import, version, CPUExecutionProvider present, providers non-empty, session creation |
| `test_ort_inference.py` | ReLU, MatMul+Bias, Softmax correctness; zero boundary; negative-only; determinism; random inputs |

### OVEP Suite (`tests/ovep/`) — Release & RelWithDebInfo only

| File | Tests |
|---|---|
| `test_ovep_provider.py` | OVEP in providers list, ORT version, session creation, input names |
| `test_ovep_inference.py` | ReLU, MatMul+Bias, Softmax via OVEP; random inputs via OVEP |
| `test_ovep_vs_cpu.py` | ReLU / Softmax / MatMul+Bias parity (OVEP vs CPU); 5 random inputs; softmax sum preserved at various scales |

---

## Build Type vs Test Suite Matrix

| Build Type | OV suite | ORT suite | OVEP suite |
|---|---|---|---|
| Release | ✅ All tests | ✅ All tests | ✅ All tests |
| RelWithDebInfo | ✅ All tests | ✅ All tests | ✅ All tests |
| Debug | ⛔ Skipped (no Python bindings) | ✅ All tests | ⛔ Skipped |

---

## Environment Variables (set by run_tests.py)

| Variable | Description |
|---|---|
| `OV_INSTALL_DIR` | Full path to the OV install tree, e.g. `C:\ov_build\install\openvino\Release` |
| `BUILD_TYPE` | CMake config: `Release`, `RelWithDebInfo`, or `Debug` |
| `OV_DEVICE` | OpenVINO device string: `CPU` or `GPU` |
| `ORT_DEVICE` | Raw device for ORT OVEP: `CPU` (mapped to `CPU_FP32`) or `GPU` (→ `GPU_FP32`) |
| `PYTHONPATH` | Prepended with `<OV_INSTALL_DIR>\python` and `<OV_INSTALL_DIR>\python\python3` |
| `OPENVINO_LIB_PATHS` | Semicolon-separated list of OV runtime DLL directories |
| `PATH` | Prepended with OV runtime DLL directories so DLLs are loadable |

---

## Expected Output (Release, CPU, all suites)

```
================================================================
  Suite      : ALL
  Build type : Release
  OV device  : CPU
  ORT device : CPU
  OV install : C:\ov_build\install\openvino\Release
================================================================

tests/ov/test_ov_import.py .....          [ 5 passed ]
tests/ov/test_ov_inference.py .....       [ 5 passed ]
tests/ov/test_ov_dynamic_shapes.py ....   [ 4 passed ]
tests/ort/test_ort_import.py .....        [ 5 passed ]
tests/ort/test_ort_inference.py .......   [ 7 passed ]
tests/ovep/test_ovep_provider.py ....     [ 4 passed ]
tests/ovep/test_ovep_inference.py ....    [ 4 passed ]
tests/ovep/test_ovep_vs_cpu.py .....      [ 5 passed ]

======================== 39 passed in Xs =========================
```

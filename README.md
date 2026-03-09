# OpenVINO Multi-Agent Build System

Automated multi-agent pipeline that clones, builds, installs, and verifies
**OpenVINO** and **ONNX Runtime with OpenVINO Execution Provider (OVEP)** on Windows.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Proxy Configuration](#proxy-configuration)
3. [User Inputs](#user-inputs)
4. [Usage](#usage)
5. [Pipeline Overview](#pipeline-overview)
6. [Build Instructions](#build-instructions)
   - [Full pipeline (OpenVINO + ORT)](#full-pipeline-openvino--ort)
   - [ORT-only (OpenVINO already built)](#ort-only-openvino-already-built)
   - [Verify OpenVINO only](#verify-openvino-only)
7. [Configuration Reference](#configuration-reference)
8. [Directory Layout](#directory-layout)
9. [Output Artifacts](#output-artifacts)
   - [OpenVINO wheels](#openvino-wheels)
   - [OpenVINO runtime binaries](#openvino-runtime-binaries)
   - [ONNX Runtime wheels](#onnx-runtime-wheels)
   - [OVEP install (for ORT GenAI)](#ovep-install-for-ort-genai)
10. [Build Report](#build-report)
11. [Platform Notes](#platform-notes)

---

## Prerequisites

| Tool | Minimum version | Notes |
|---|---|---|
| Python | 3.10+ | Must be on `PATH`; venv is created automatically |
| CMake | 3.25+ | Must be on `PATH` |
| Visual Studio | 2022 (VS 17) or 2026 (VS 18) | Desktop C++ workload required |
| Git | any recent | Must be on `PATH` |

> **Proxy:** If your machine is behind a corporate proxy, set `http_proxy` /
> `https_proxy` environment variables before running. The system defaults to
> `http://proxy-us.intel.com:911` when neither variable is set.

---

## Proxy Configuration

By default the system makes **no assumption about a proxy** and connects
directly. If your network requires a proxy, set the standard environment
variables **before** running `main.py`:

```powershell
# PowerShell
$env:http_proxy  = "http://proxy.example.com:911"
$env:https_proxy = "http://proxy.example.com:911"

# Then run as normal
python main.py C:\ov_build Release RelWithDebInfo
```

```cmd
REM Command Prompt
set http_proxy=http://proxy.example.com:911
set https_proxy=http://proxy.example.com:911
python main.py C:\ov_build Release RelWithDebInfo
```

The values are automatically forwarded to:
- `git clone` (via `http.proxy` git config)
- `pip install` (venv bootstrap and wheel installs)
- CMake / ORT builds (via environment inheritance)

To make the setting persistent across sessions, add the variables to your
system or user environment (Windows → System Properties → Environment Variables).

---

## User Inputs

### Positional arguments

| Argument | Description | Example |
|---|---|---|
| `base_directory` | Root workspace where sources, builds, and installs are placed | `C:\ov_build` |
| `build_types` *(optional)* | Space-separated list of CMake configs to produce | `Release RelWithDebInfo` |

If `build_types` is omitted, the default is **`Release RelWithDebInfo Debug`**.

### Optional flags

| Flag | Description |
|---|---|
| `--ort-only` | Skip the OpenVINO phase entirely; build & verify ORT assuming OV is already installed |
| `--verify-ov-only` | Run only the OpenVINO verification tests (no builds) |

### Configuration (`config.py`)

Advanced settings can be changed directly in [config.py](config.py):

| Field | Default | Description |
|---|---|---|
| `repo_url` | `https://github.com/openvinotoolkit/openvino.git` | OpenVINO git remote |
| `repo_branch` | `master` | OpenVINO branch |
| `cmake_generator` | `Visual Studio 18 2026` | CMake generator string — change to `Visual Studio 17 2022` for VS 2022 |
| `cmake_extra_args` | see below | Extra `-D` flags forwarded to CMake |
| `ort_repo_url` | `https://github.com/intel/onnxruntime.git` | ORT-OVEP fork remote |
| `ort_branch` | `ovep-develop` | ORT branch with OVEP support |
| `ort_build_types` | `["Release", "RelWithDebInfo"]` | Which configs to build for ORT |
| `ort_openvino_devices` | `["CPU", "GPU"]` | OVEP target devices |
| `build_timeout` | `7200` (2 h) | Per-config build timeout in seconds |
| `ort_build_timeout` | `7200` (2 h) | ORT build timeout in seconds |
| `parallel_jobs` | `8` | `-j` value passed to ninja / msbuild |

Default `cmake_extra_args` passed to the OpenVINO CMake configure step:

```
-DENABLE_DEBUG_CAPS=ON
-DENABLE_PYTHON=ON
-DENABLE_WHEEL=ON
-DENABLE_TESTS=ON
-DENABLE_INTEL_GPU=ON
```

> **Note:** `-DENABLE_PYTHON` and `-DENABLE_WHEEL` are automatically disabled for
> `Debug` builds because the OpenVINO Python bindings cannot be compiled in Debug
> mode on Windows.

---

## Usage

```powershell
# Full pipeline — all 3 configs (Release, RelWithDebInfo, Debug)
python main.py C:\ov_build

# Full pipeline — specific configs only
python main.py C:\ov_build Release
python main.py C:\ov_build Release RelWithDebInfo

# ORT+OVEP phase only (OpenVINO already built/installed)
python main.py C:\ov_build Release RelWithDebInfo --ort-only

# OpenVINO verification only (no builds)
python main.py C:\ov_build Release RelWithDebInfo --verify-ov-only
```

The script bootstraps a virtual environment at `.venv/` on first run and
re-executes itself inside it automatically. Subsequent runs reuse the venv.

---

## Pipeline Overview

```
Phase 1 — OpenVINO
  ├─ DownloadAgent     Clone openvinotoolkit/openvino (once)
  └─ per build type:
       ├─ BuildAgent       cmake configure + cmake --build
       ├─ InstallAgent     cmake --install
       └─ VerifyAgent      3 Python tests (import, Core, inference)

Phase 2 — ONNX Runtime + OVEP
  ├─ OrtDownloadAgent  Clone intel/onnxruntime (once)
  └─ per ort_build_type × device (CPU, GPU):
       ├─ OrtBuildAgent    build.bat --use_openvino …
       ├─ OrtInstallAgent  pip install wheel, generate ovep_install
       └─ VerifyAgent      3 OVEP tests (provider presence, inference, CPU vs OVEP)
```

---

## Build Instructions

### Full pipeline (OpenVINO + ORT)

```powershell
python main.py <base_dir> [Release] [RelWithDebInfo] [Debug]
```

**What happens:**

1. Git clone of OpenVINO into `<base_dir>\openvino_source`
2. For each build type:
   - CMake configure with Visual Studio generator + Python bindings enabled
   - `cmake --build` (uses MSBuild with `--parallel`)
   - `cmake --install` into `<base_dir>\install\openvino\<BuildType>\`
   - Python verification tests run against the install
3. Git clone of ORT (`ovep-develop` branch) into `<base_dir>\onnxruntime_source`
4. For each ORT build type × device (`CPU`, `GPU`):
   - `build.bat` invoked with `--use_openvino` and `OpenVINO_DIR` pointing at the matching OV install
   - Python wheel installed via `pip`
   - `cmake -P cmake_install.cmake` generates the `ovep_<type>_<device>` directory
   - OVEP inference correctness tests run

**OpenVINO CMake command (generated internally):**

```
cmake <source_dir> -B <build_dir>
  -G "Visual Studio 18 2026"
  -DCMAKE_BUILD_TYPE=<BuildType>
  -DCMAKE_INSTALL_PREFIX=<install_dir>
  -DENABLE_DEBUG_CAPS=ON
  -DENABLE_PYTHON=ON
  -DENABLE_WHEEL=ON
  -DENABLE_TESTS=ON
  -DENABLE_INTEL_GPU=ON
  -DPython3_EXECUTABLE="<path_to_python>"
```

**ORT build command (generated internally):**

```
build.bat
  --config <BuildType>
  --cmake_generator "Visual Studio 18 2026"
  --use_openvino
  --build_wheel
  --build_shared_lib
  --parallel
  --skip_tests
  --build_dir "<ort_build_dir>"
  --cmake_extra_defines OpenVINO_DIR="<ov_install>\runtime\cmake"
```

### ORT-only (OpenVINO already built)

Use when OpenVINO has already been built and installed in the expected layout.
The system skips Phase 1 entirely and jumps straight to the ORT build.

```powershell
python main.py C:\ov_build Release RelWithDebInfo --ort-only
```

### Verify OpenVINO only

Runs the 3 Python verification tests against each requested build type's install
without triggering any builds. Useful for a quick sanity-check after installing.

```powershell
python main.py C:\ov_build Release RelWithDebInfo --verify-ov-only
```

---

## Configuration Reference

### Changing the Visual Studio version

Edit `cmake_generator` in `config.py`:

```python
# VS 2022
cmake_generator: str = "Visual Studio 17 2022"

# VS 2026 (default)
cmake_generator: str = "Visual Studio 18 2026"
```

### Changing the ORT branch or remote

```python
ort_repo_url: str = "https://github.com/intel/onnxruntime.git"
ort_branch: str = "ovep-develop"
```

### Limiting ORT build configs or devices

```python
ort_build_types: List[str] = ["Release"]          # Release only
ort_openvino_devices: List[str] = ["CPU"]          # CPU only
```

---

## Directory Layout

All output is placed under `<base_dir>`:

```
<base_dir>\
├── openvino_source\          OpenVINO git clone
├── onnxruntime_source\       ORT git clone
│
├── build\
│   ├── openvino\
│   │   ├── Release\          OV CMake build tree (Release)
│   │   ├── RelWithDebInfo\   OV CMake build tree (RelWithDebInfo)
│   │   └── Debug\            OV CMake build tree (Debug)
│   └── onnxruntime\
│       ├── Release\
│       │   ├── CPU\          ORT build tree (Release, CPU)
│       │   └── GPU\          ORT build tree (Release, GPU)
│       └── RelWithDebInfo\
│           ├── CPU\
│           └── GPU\
│
├── install\
│   ├── openvino\
│   │   ├── Release\          OV install tree (Release)
│   │   ├── RelWithDebInfo\   OV install tree (RelWithDebInfo)
│   │   └── Debug\            OV install tree (Debug)
│   └── onnxruntime\
│       ├── Release\
│       │   ├── CPU\          ORT install tree (Release, CPU)
│       │   └── GPU\          ORT install tree (Release, GPU)
│       └── RelWithDebInfo\
│           ├── CPU\
│           └── GPU\
│
├── build_report.json         Pipeline results (updated after every run)
└── build_agent.log           Detailed log of the run
```

---

## Output Artifacts

### OpenVINO wheels

Built only for `Release` and `RelWithDebInfo` (Python bindings are disabled in
`Debug`).

```
<base_dir>\build\openvino\<BuildType>\wheels\
    openvino-<version>-<pyver>-<pyver>-win_amd64.whl
```

Example:
```
C:\ov_build\build\openvino\Release\wheels\
    openvino-2026.1.0-21257-cp314-cp314-win_amd64.whl
```

### OpenVINO runtime binaries

After `cmake --install`, the layout under each build type is:

```
<base_dir>\install\openvino\<BuildType>\
├── runtime\
│   ├── bin\
│   │   └── intel64\
│   │       └── <BuildType>\          *.dll — core runtime DLLs
│   ├── cmake\                         OpenVINOConfig.cmake (used by ORT build)
│   ├── include\                       C++ headers
│   ├── lib\                           *.lib import libraries
│   └── 3rdparty\
│       └── tbb\                       TBB redistribution
├── python\                            openvino Python package
│   └── python3\
└── tools\
```

Key paths mirrored by `setupvars.bat`:

| Purpose | Path |
|---|---|
| Runtime DLLs | `install\openvino\<BuildType>\runtime\bin\intel64\<BuildType>\` |
| TBB DLLs | `install\openvino\<BuildType>\runtime\3rdparty\tbb\redist\intel64\vc14\` |
| Python package | `install\openvino\<BuildType>\python\` |
| CMake config | `install\openvino\<BuildType>\runtime\cmake\` |

### ONNX Runtime wheels

```
<base_dir>\build\onnxruntime\<BuildType>\<Device>\Windows\<BuildType>\dist\
    onnxruntime_openvino-<version>-<pyver>-<pyver>-win_amd64.whl
```

Example:
```
C:\ov_build\build\onnxruntime\Release\CPU\Windows\Release\dist\
    onnxruntime_openvino-1.25.0-cp314-cp314-win_amd64.whl
```

The wheel is automatically installed into the active Python environment by
`OrtInstallAgent` after each successful build.

### OVEP install (for ORT GenAI)

A reorganized install tree is generated alongside each ORT build for use by
downstream ORT GenAI builds:

```
<base_dir>\build\onnxruntime\<BuildType>\<Device>\Windows\<BuildType>\
    ovep_<buildtype_lower>_<device_lower>\
        include\          onnxruntime C++ headers (flattened)
        lib\              *.dll + *.lib
```

Example:
```
C:\ov_build\build\onnxruntime\Release\CPU\Windows\Release\
    ovep_release_cpu\
        include\
        lib\
```

The install is generated by running `cmake -P cmake_install.cmake` inside the
ORT Windows build directory and then reorganizing `bin/` → `lib/` and
`include/onnxruntime/` → `include/`.

---

## Build Report

After every run `<base_dir>\build_report.json` is written with a per-stage
pass/fail summary, error messages, and timing. Example structure:

```json
{
  "timestamp": "2026-03-09T09:50:47",
  "total_duration_human": "2.8 minutes",
  "results": {
    "ort_build_Release_CPU":   [{ "success": true,  "message": "ORT-OVEP build succeeded [CPU] in 31s" }],
    "ort_install_Release_CPU": [{ "success": true,  "message": "ORT installed [CPU] (OVEP=YES)" }],
    "verify_ovep_Release_CPU": [{ "success": false, "errors": ["..."] }]
  }
}
```

A human-readable log is also written to `<base_dir>\build_agent.log`.

---

## Platform Notes

| Topic | Detail |
|---|---|
| **OS** | Windows only. The pipeline uses `build.bat`, Visual Studio generators, and `.exe` paths. |
| **Python bindings in Debug** | OpenVINO's Python extension cannot be compiled in Debug mode on Windows. `ENABLE_PYTHON` and `ENABLE_WHEEL` are auto-disabled for Debug builds. |
| **ORT devices** | Both `CPU` and `GPU` OVEP variants are built by default. GPU requires an Intel GPU with up-to-date drivers. Remove `"GPU"` from `ort_openvino_devices` in `config.py` to skip it. |
| **Virtual environment** | `main.py` auto-creates `.venv/` next to itself and re-executes inside it. All `pip install` operations during install and verify use the same venv Python (`sys.executable`). |
| **Proxy** | Set `http_proxy` / `https_proxy` before running if behind a firewall. The value is forwarded to `git`, `pip`, and `cmake` automatically. See [Proxy Configuration](#proxy-configuration) below. |

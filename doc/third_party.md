# Third-Party Dependencies

This document records third-party libraries incorporated into or used by `Inferlet`.

## 1. llama.cpp

- **Repository**: `https://github.com/ggml-org/llama.cpp.git`
- **Location**: `third_party/llama.cpp` (Git Submodule)
- **Pinned Commit**: `8b4b3558f1459c13e4aa38d5c94d306a00dc6acd`
- **Commit Date**: 2026-09-04
- **License**: MIT License (see `third_party/llama.cpp/LICENSE`)
- **Integration Policy**:
  - Pinned to exact 40-character commit hash. Automatic tracking of master is strictly prohibited.
  - Subproject integration via `add_subdirectory(third_party/llama.cpp EXCLUDE_FROM_ALL)`.
  - Unnecessary subproject targets (tools, examples, server, tests, apps, Web UI) are disabled in `cmake/InferletOptions.cmake`.
  - Offline build only; network fetching during build is prohibited.

## 2. nlohmann/json

- **Location**: `third_party/llama.cpp/vendor/nlohmann/json.hpp`
- **License**: MIT License
- **Usage**: Used internally for parsing `inferlet_task_input_set_tools` and JSON run configuration files. Not exposed in public ABI headers.

## 3. libmtmd and its bundled dependencies

- **Location**: `third_party/llama.cpp/tools/mtmd`, same pinned llama.cpp commit.
- **Usage**: CPU image preprocessing, vision encoding/projection and multimodal decoding; bundled into `libinferlet.a` with `vendor-hash`.
- **Dependencies**: stb (public domain/MIT), miniaudio (public domain/MIT-0), sheredom (Unlicense), and hash implementations (xxHash BSD-2-Clause, SHA-1 public domain, SHA-256 public domain, rotate-bits MIT). License texts are retained in the upstream sources; installed notices are under `share/licenses/Inferlet`.
- Upstream mtmd tools and video subprocess helpers are disabled. The upstream library includes its registered model implementations, including audio architectures; Inferlet exposes image input only.

# 接口、构建与外部集成审计

## Inferlet 命名统一

项目文件、文档、注释、C API、C++ 命名空间、CMake 配置、pkg-config 和发布流程已统一使用 Inferlet。公共头文件为 `include/inferlet.h`，64 个公开函数使用 `inferlet_*` 小写下划线命名，类型使用相同前缀，常量和环境变量使用 `INFERLET_*`。CMake 包及目标为 `Inferlet` / `Inferlet::Inferlet`，pkg-config 包为 `inferlet`，静态库为 `libinferlet.a`。

公开符号已更名，调用方需要更新标识符、包含路径和链接配置并重新编译。没有保留旧名称别名；参数、结构体布局和错误码数值保持不变。新库的符号检查确认 64 个声明和导出一一对应，没有旧名称符号。

以新构建目录 `build_inferlet` 完成 Release 验证：**27/27** 测试通过，无跳过，耗时 **67.07 秒**。覆盖真实文本和图像推理、错误路径回归、C/C++ 消费者、CMake 和 pkg-config 安装。112 个项目文件的文件名和内容扫描均无旧名称，文档本地链接检查通过；发布工作流 YAML 解析通过。

本机 Linux aarch64 产物为 `lib/libinferlet.a`、`lib/arm/libinferlet.a`，完整安装包位于 `build_inferlet/install/`。此前的本地库归档在 `build_legacy_artifacts/`，x64/macOS 新产物需由相应平台构建或更新后的 CI 生成。日志和符号报告位于 `build_inferlet/audit/`。

以下保留命名调整前各阶段的验证记录，文档术语已按当前名称统一。

## 错误路径复验与修复（历史记录）

本次复验发现流式收尾、调度器和 `inferlet_get_stream_text` 用静态工厂 `Status::ok()` 判断当前状态，导致错误丢失；Tokenizer 解码越界 token 则把底层异常误报为内部错误。现已修复，并补充可检出原问题的回归测试。

- 流式失败和取消保留原始错误，收尾错误会传递到任务终态及内部 last_error。现有流式测试中的恒真 `assert(st.ok())` 改为实际状态检查。
- 同步、异步及 beam search 的收集结果在交付前检查 UTF-8 完整性；生成上限截断多字节字符时，返回 `INFERLET_E_INFERENCE`，不会交付不完整文本。流式同样记录失败，并保留已交付的完整 chunk。
- `inferlet_tokenizer_decode` 先校验全部 token ID。负数、词表上界及更大 ID 返回 `INFERLET_E_INVALID_ARGUMENT`，输出长度为 0，缓冲区保持不变；长度查询模式遵循同样规则。
- 更新公共头文件中的视觉支持说明、查询错误码和流式结束语义。公共函数仍为 64 个，函数声明和 ABI 导出不变。视频、预计算视觉特征、slow tokenizer 等既有功能限制未扩展。

修复前，`unit_result_sink`、`integration_tokenizer`、`integration_error_paths` 三项回归均失败；修复后全部通过。Release 完整测试 **27/27** 通过，无跳过，耗时 **66.66 秒**，包含 Qwen 文本模型、SmolVLM 真实图像、C/C++ 及 CMake/pkg-config 外部集成。独立编译的结果收集单元测试通过 ASan/UBSan（含泄漏检测），没有 sanitizer 报告；本次未重新运行全库 sanitizer。

复验日志位于 `build_verify/current-audit/`：`regressions-before.log`、`regressions-after.log`、`fix-ctest.log`、`fix-sink-asan.log`、`fix-abi.json`。构建产物为 Linux aarch64 CPU Release，交付到 `lib/libinferlet.a` 和 `lib/arm/libinferlet.a`；其他架构的旧二进制未重建。

测试模型目录已迁移，测试可通过 `INFERLET_TEST_MODEL` 指定文本模型，`integration_models_folder` 另用 `INFERLET_TEST_MODEL_DIR` 指定包含两个 Qwen 模型的目录；图像环境变量沿用 [vision.md](vision.md)。

流式 ABI 仍只返回文本或 NULL，没有公开错误查询函数。需要明确完成错误码的调用方应使用 `inferlet_run` 或 `inferlet_run_async` 配合 `inferlet_wait_task`/`inferlet_query_task`；本次修复没有新增公共接口。

以下保留此前的审计与图像接入记录。

审计日期：2026-09-05。环境：Linux aarch64、GCC/G++ 13.3、CMake，CPU 后端。
llama.cpp 子模块：`8b4b3558f1459c13e4aa38d5c94d306a00dc6acd`。

## 历史审计结论

`include/inferlet.h` 的 64 个函数均有实现，静态库导出的 64 个 `inferlet_*` 符号与头文件逐一一致，公共 C ABI 未改变。
已实现的文本推理功能通过真实模型测试，但不能把“有符号”和“拒绝不支持的请求”当作所有功能已经完成。
审计修复已提交为 `5283f77`。随后按用户要求接通 CPU 图像推理，配置和验证方法见 [图像集成说明](vision.md)。仍有明确不支持的配置和输入格式。

## 功能核对

| 接口范围 | 核对结果与边界 |
|---|---|
| Init / inferlet_deinit / inferlet_load_from_file / Unload / inferlet_check_backend_active | 初始化引用计数、模型共享、同步/异步加载、失败处理与忙闲查询已有实现；同一 handle 不能跨线程同时使用。 |
| Run / inferlet_run_async / inferlet_run_stream / inferlet_wait_task / inferlet_query_task / inferlet_cancel_task | 同步、异步、流式、查询、等待和取消经过真实模型测试；同步调用现已遵守共享 backend 数量限制。 |
| TaskInput 分配、释放；PromptText / PromptTokens / ChatHistory | 参数验证、深拷贝、输入类型覆盖和提交快照已有实现及测试。 |
| TopK / TopP / Temperature / RepetitionPenalty / MaxOutputTokens / ProbabilityType | 采样与概率输出已有实现；非有限浮点参数现返回 `INFERLET_E_INVALID_ARGUMENT`，并保留原有效配置。 |
| BeamSize | 已实现 beam search，包含确定性、guided choices 和取消测试；beam > 1 不支持输出 logits/embedding。 |
| VerboseDataFlag；TaskOutput 和 Stream 的全部 getter | 测试覆盖文本、tokens、字节偏移、概率、输入 tokens、logits、embedding、指针稳定性和流式分块；logits 是原始分数，并非归一化概率。 |
| StreamBlockMode / SpecialTokenMode | 阻塞/非阻塞读取、特殊 token 文本保留规则已有实现和测试。 |
| GuidedChoices / Tools | grammar 限制和工具描述注入已有实现；两者不能同时使用，原始 token 输入不支持 Tools。工具执行本身由调用方负责。 |
| LowRankAdapterId / inferlet_parameter_set_low_rank_adapter | 注册、加载、选择和并发隔离有测试，使用测试生成的 LoRA GGUF；不等价于任意生产 adapter 的兼容性认证。 |
| ThinkingMode | 支持模型的默认思考输出有测试；关闭 Qwen3 系列思考模式会返回 `INFERLET_E_UNSUPPORTED`。 |
| SetMultiModal / SetMultiModal_ChatHistory | ChatHistory 的原始 UINT8 RGB 图像通过真实 libmtmd 编码、投影和解码，支持同步/异步/流式与特征缓存。legacy 和预计算特征、视频仍明确拒绝；未配投影器时返回不支持。 |
| VisualEncoder / VisionMaxPatchCnt / VisionMaxGridSize / VisionTransSkipped | VisualEncoder 支持空值/default；其他编码器 ID 拒绝。自定义 patch/grid 限制及跳过视觉编码仍不支持，保留默认值。 |
| Parameter 分配、释放；TokenizerOnly / DecoderMaxTokenCnt / LoadFromFileSync / RunConfigFile | tokenizer-only、batch、加载模式、JSON 配置和限制已有实现和测试；JSON `scheduler.worker_count` 非零不受支持。 |
| TokenizerSlow | 非默认 slow 模式不支持，加载返回 NULL。 |
| DeviceId / DeviceList / BackendCnt | 设备枚举和选择、列表校验、执行名额限制已有实现；本次只验证 CPU，未验证多 GPU 分布式推理。 |
| inferlet_tokenizer_encode / inferlet_tokenizer_decode | UTF-8 往返、特殊 token、缓冲区不足与截断已有测试；Decode 缓冲区还需预留字符串终止符。 |

流式错误可观察性仍有限：`inferlet_get_stream_text` 对正常结束和执行失败都返回 NULL，详细错误存放在内部 last_error/日志，头文件没有公开错误查询接口。本次没有扩展 ABI。
默认模型加载是异步的，`inferlet_load_from_file` 返回非空仅表示 handle 创建成功；应使用同步加载配置或检查后续运行/状态查询结果。

## 本次最小修正

1. **执行名额和忙闲统计不一致。** 原来同步 `inferlet_run` 绕过 backend 限制，其他共享 handle 查询到的状态始终可能是空闲；异步任务完成但未取结果时又错误地显示忙。加入同步执行名额的 RAII 管理及计数，忙闲状态只依据加载/排队/执行工作，并分别添加回归测试。输出对象使用 RAII，防止异常路径泄漏。
2. **无效浮点参数被当作成功。** TopP 接受 NaN，Temperature 和 RepetitionPenalty 接受 NaN/无穷值。现拒绝这些输入，测试同时验证原有效设置不被覆盖。
3. **关闭 OpenMP 后仍强加依赖。** `GGML_OPENMP=OFF` 时不再链接或向 CMake/pkg-config 使用方传播 OpenMP。
4. **pkg-config 安装路径不可迁移。** 原 `.pc` 固定指向 `/usr/local`，安装测试手工覆盖 prefix 掩盖了问题。现根据 `pcfiledir` 计算 prefix，测试直接使用安装后的 `.pc`。
5. **含空格路径无法合并静态库。** GNU ar MRI 不识别这种路径（加引号也无效）。现先暂存为简单文件名再合并；缺失依赖直接失败，使用 CMake 工具链指定的 ar/ranlib，合并脚本变动会触发重新打包。
6. **关闭 sanitizer 后仍残留第三方编译选项。** 同一构建目录 ASan ON → OFF 原来只关闭 wrapper 和导出选项，llama 仍被插桩。现在同步重置上游缓存选项。

外部 C 测试补充 tokenizer、同步、异步、流式及提交后释放输入；显式指定测试模型时，C++ consumer 不再将加载失败当成成功。

## 构建必要性

- 默认目标是一个 STATIC 库，归档包含 Inferlet、mtmd、vendor-hash、llama、ggml、ggml-base、ggml-cpu；使用方无需另外获取这些第三方静态库或头文件。
- 上游 common、工具、server、app、UI、示例、测试均已关闭；项目自身测试和示例默认关闭。视觉功能在审计提交后按用户要求单独接入，使用现有 libmtmd 库目标。
- C++17 用于内部实现；C11 可使用公共接口。PIC 使该静态库可被共享库引用。
- llama 编译多个模型架构是当前通用模型加载器的组成部分；直接删源文件可能破坏其架构工厂引用，因此没有为减少编译数量而裁剪。
- 此次图像接入替换了原 mtmd 占位计算并补齐实际投影器加载、模板、预处理、编码及解码链路；`MTMD_VIDEO=OFF`，无需 ffmpeg。其余未使用的空实现文件不补入构建。
- 分发 CPU 静态库建议显式 `GGML_NATIVE=OFF`，避免为构建机器生成专用指令。OpenMP 是可选的 CPU 并行依赖。
- 本次未验证 Windows/macOS、GPU 后端或跨架构产物；静态库应与使用方的架构、工具链和 C++ 运行时匹配。
- 归档合并清单固定为 CPU 依赖。开启 CUDA/Vulkan/BLAS 等额外后端后，还需处理其归档和系统依赖，不能直接套用本次两文件集成的结论。

## 复现与集成

审计阶段 Release 全量测试 23/23 通过；无 OpenMP 全量测试 23/23 通过，最后的忙闲修正另通过相关 15/15 回归。
审计阶段 ASan/UBSan 全量测试 23/23 通过，未报告地址错误、泄漏或未定义行为；可选返回后栈检查的范围限制见下文。
只复制归档和公共头文件的 C/C++ 程序也通过真实模型推理；C 程序还覆盖 tokenizer、异步和流式调用。
使用了 Qwen2.5-0.5B Q4_K_M 和 Qwen3.5-0.8B Q4_K_M；这些结果不代表所有 GGUF 模型或硬件均已认证。

日志保存在 `build_verify/audit_logs/`。`lib/libinferlet.a` 已更新为本次验证的普通 Release 产物：Linux aarch64、CPU、`GGML_NATIVE=OFF`、OpenMP 开启，不含 sanitizer 依赖。
公共头文件仍为 `include/inferlet.h`，完整 CMake/pkg-config 安装包位于 `build_verify/install/`。

```sh
cmake -S . -B build_verify -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=OFF -DINFERLET_BUILD_TESTS=ON -DINFERLET_BUILD_EXAMPLES=ON
cmake --build build_verify -j4
INFERLET_TEST_MODEL="$PWD/models/qwen2.5-0.5b-instruct-q4_k_m.gguf" \
  ctest --test-dir build_verify --output-on-failure --timeout 300
cmake --install build_verify --prefix "$PWD/build_verify/install"
```

内存检查复现命令：

```sh
cmake -S . -B build_verify_asan -DCMAKE_BUILD_TYPE=Debug \
  -DINFERLET_BUILD_TESTS=ON -DINFERLET_ENABLE_ASAN=ON \
  -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF
cmake --build build_verify_asan -j4
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1:detect_stack_use_after_return=0 \
UBSAN_OPTIONS=halt_on_error=1 \
INFERLET_TEST_MODEL="$PWD/models/qwen2.5-0.5b-instruct-q4_k_m.gguf" \
  ctest --test-dir build_verify_asan --output-on-failure --timeout 300
```

本机开启 ASan 的可选返回后栈检查时，加载期间在 `__asan::FakeStack::Allocate` 出现显著开销，单个生命周期测试耗时约 177 秒后通过。
上述命令关闭该可选项，仍检查地址错误、释放后使用、泄漏和未定义行为；不覆盖函数返回后的栈访问。没有运行 TSan 动态检测，仅验证了其编译配置切换。

部分旧集成测试仍使用本机默认模型路径，`integration_models_folder` 还需要 Qwen3.5 模型，LoRA 测试需要 Python numpy/gguf。移到其他机器运行这些测试前需准备相应测试数据；发布库运行不依赖 Python。

使用安装包时，C/C++ 项目均可：

```cmake
find_package(Inferlet CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE Inferlet::Inferlet)
```

使用 pkg-config 时必须包含静态库的系统依赖：

```sh
export PKG_CONFIG_PATH="$PWD/build_verify/install/lib/pkgconfig"
gcc consumer.c $(pkg-config --cflags --libs --static inferlet) -o consumer
```

只复制 `include/inferlet.h` 和 `lib/libinferlet.a` 时，本次 Linux/GCC、启用 OpenMP 的产物可按以下方式链接：

```sh
gcc -std=c11 -Iinclude consumer.c lib/libinferlet.a \
  -lstdc++ -fopenmp -pthread -ldl -lm -o consumer
g++ -std=c++17 -Iinclude consumer.cpp lib/libinferlet.a \
  -fopenmp -pthread -ldl -lm -o consumer_cpp
```

对于 `GGML_OPENMP=OFF` 的产物可移除 `-fopenmp`。不要把 ASan/TSan 测试产物替代正常发布库；它们需要相应 sanitizer 运行时。
静态库内含 llama/ggml 全局符号，不能假定可以与另一版本的 llama/ggml 任意混链。

## 图像接通后的验证

在提交审计修复 `5283f77` 后接入真实 libmtmd，未更改 llama.cpp 子模块版本或公共 ABI。

- Release 全量测试 **24/24** 通过。追加多图聊天历史和配置校验后，相关 ABI/配置/图像测试 **7/7** 通过。
- 最终归档的 ABI 符号、配置、CMake/pkg-config consumer 检查 **3/3** 通过。
- ASan/UBSan（OpenMP OFF）的配置、同步、异步、流式、真实图像测试 **5/5** 通过；图像测试包含多图历史，耗时 179.78 秒。未报告地址错误、泄漏或未定义行为；仍使用前文说明的 `detect_stack_use_after_return=0`，不覆盖该可选检查。
- 仅复制头文件和归档，用 `gcc -std=c11 -Wall -Wextra -Wpedantic -Werror` 编译的外部 C 图像测试通过：红图回答 `Red.`、蓝图回答 `Blue.`，带前轮红图的多图历史回答最后一张图为 `Blue.`；同步/异步/流式一致，覆盖输入释放、缓存冲突、错误尺寸、未知编码器、图像 beam 拒绝和取消。
- 投影文件不存在、Qwen2.5 文本模型配 SmolVLM 投影器两种负向加载探针均返回 NULL，没有崩溃。
- 视觉测试资产为 SmolVLM-256M-Instruct Q8_0 与配套 mmproj；图像由测试生成，不需要外部图片或图片解码依赖。其他视觉模型、GPU 图像编码、视频和预计算特征不在本次验证范围内。

图像阶段日志与测试资产位于 `build_vision/`。交付归档 SHA-256：
`5b5e5785577dc52ad8f4b13f9edb3cd7f0789b0f8647b904f84b363e8c3a7c09`。

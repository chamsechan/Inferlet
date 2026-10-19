# Inferlet 实现计划

> 本文是里程碑路线图。逐文件、逐 API、逐流程的落地说明见
> [`inferlet_detailed_design.md`](./inferlet_detailed_design.md)。

## 1. 目标与范围

本项目以 `llama.cpp` 为推理内核，对外只暴露 `include/inferlet.h` 中的 C ABI。构建产物至少包括：

- `include/inferlet.h`
- `lib/libinferlet.a`
- 推荐同时提供 `lib/cmake/Inferlet/InferletConfig.cmake` 和 `lib/pkgconfig/inferlet.pc`，用于传递静态库仍需链接的系统库、GPU 运行库和编译选项

调用方不应直接依赖任何 llama.cpp 类型。所有 `llama_model`、`llama_context`、sampler、KV cache、视觉编码器和线程对象都必须隐藏在实现内部。

建议按“文本同步 MVP -> 异步与流式 -> 高级采样和 LoRA -> 多模态”的顺序实现。不要一开始同时实现全部 64 个函数；先固定 API 契约和生命周期，否则后续很容易出现 ABI 不变但行为互不兼容的问题。

## 2. 首先要冻结的 API 契约（P0）

当前头文件给出了函数签名，但下列语义不完整或互相矛盾。应先写入 `doc/api_contract.md`，并把相关说明同步回头文件注释，再开始实现。

| 问题 | 建议约定 |
| --- | --- |
| C 兼容性 | 当前结构体只有 `struct inferlet_chat_history` 声明，函数参数却使用 `inferlet_chat_history`，因此头文件只能按 C++ 编译。若确实需要 C ABI/C 调用方，应改成 `typedef struct ... ...;`；若只支持 C++ 调用方，则文档明确这一点。此修改不改变二进制布局。 |
| 头文件保护宏 | `__INFERLET_H__` 属于实现保留命名，改为 `INFERLET_H_INCLUDED`。 |
| 导出宏 | `IVSAPI_EXPORT` 与项目名不一致。静态库暂不受影响，但建议换成 `INFERLET_BUILDING_LIBRARY`，并同时补齐 Windows DLL 宏，为未来共享库留余地。 |
| 错误码 | 公开稳定的负错误码：非法参数、非法句柄、错误状态、加载失败、推理失败、不支持、已取消、资源不足、内部错误。所有导出函数禁止让 C++ 异常越过 ABI。当前没有 `GetLastError`，首版至少将详细原因写日志。更推荐新增线程局部的 `inferlet_get_last_error()`。 |
| setter 的数据所有权 | 所有 setter 深拷贝字符串、数组和特征数据。setter 返回后调用方即可释放原始数据。这样异步任务才不会悬空。超大多模态输入以后可新增显式零拷贝 API，不能暗中改变当前接口语义。 |
| 一次任务的输入类型 | prompt text、prompt tokens、普通 chat、多模态、多模态 chat 互斥；建议“最后一次内容 setter 生效并清除前一种内容”。任务提交时再做完整校验。 |
| `TaskInput` 提交后的行为 | `inferlet_run`/`inferlet_run_async`/`inferlet_run_stream` 提交时制作不可变快照；提交后可以修改或释放原 `task_in`，不影响已提交任务。一个 `task_in` 可重复提交。 |
| 普通异步任务所有权 | `inferlet_run_async` 成功后得到 task；`inferlet_query_task` 或 `inferlet_wait_task` 首次成功交付 `task_out` 时自动销毁 task。交付后禁止再次使用 task。取消后仍需调用 `inferlet_wait_task` 或轮询 `inferlet_query_task` 完成回收。 |
| 流任务所有权 | 保持头文件现有约定：`inferlet_get_stream_text()` 返回 `NULL` 时自动销毁 stream task，此后禁止再访问它。必须补充“未读取到结尾时如何释放”；建议让 `inferlet_cancel_task(handle, task_stream)` 取消并回收，或新增独立的 Stream Deallocate API。 |
| `inferlet_unload` 与在途任务 | 建议 `inferlet_unload` 停止接收新任务，取消该 handle 的所有任务，等待后台执行退出，再释放 handle。不得在 worker 仍访问 context 时直接析构。 |
| handle 线程规则 | 记录创建线程 ID，并在调试构建检查所有 handle API 是否由同一调用线程发起。后台 worker 不受此限制。多个 handle 可共享只读 model，但每个并发任务必须有独立 context/KV 状态。 |
| 模型复用 | 按 canonical model path 建立进程级 registry；同一路径第一次加载的 model 参数生效。后续参数不一致时至少打印明确告警，最好使加载失败，避免静默得到错误配置。 |
| 异步加载 | `inferlet_load_from_file` 本身没有加载结果查询接口。建议第一版无论配置如何都同步加载；正式支持 `SetLoadFromFileSync(0)` 前，先定义异步加载失败如何上报、`inferlet_run` 是否等待加载及 `inferlet_check_backend_active` 是否包括加载状态。不能静默伪装成已实现。 |
| token 范围 | 对外 `int` 与 pinned llama.cpp 版本的 `llama_token` 做 `static_assert`（大小和符号位）；所有 length 都检查非负及整数溢出。 |
| 输出 token | 明确 EOS/BOS 和其他 special token 是否进入 result token 数组。建议 result tokens 保存实际生成 token、排除终止 EOS；`SpecialTokenMode` 只控制文本是否呈现 special token。 |
| offset | 定义为 UTF-8 结果字符串中的**字节起始偏移**，数组长度与 result token 数相同。流式 offset 相对于本次 chunk，而不是累计全文。特殊 token 不呈现时，其 offset 需定义为上一文本末尾。 |
| 概率 | type=1 保存原始 logits softmax 后、任何 sampler 调整前所选 token 的概率；type=2 保存完成 penalty/temperature/top-k/top-p 等调整并重新归一化后所选 token 的概率。未开启时 getter 返回 `NULL`。 |
| logits | `GetOutputLogits` 虽然注释写“概率值”，函数名表示 logits。建议明确返回采样前的原始 logits，形状 `[generated_token_count][n_vocab]`。 |
| embedding | 定义返回每个已生成 token 经模型 decode 后的 embedding，形状 `[generated_token_count][n_embd]`。必须拷贝，不能保留 llama.cpp 临时 buffer 指针。 |
| tokenizer 截断 | encode/decode 始终先通过 llama.cpp 获取真实所需长度，写入 `real_*_length`；输出容量不足时只写可容纳部分并返回专门的 buffer-too-small 错误，避免“返回 0 但静默截断”。文本 buffer 容量包含末尾 `\0`。 |
| `beam_size` | llama.cpp 通用 sampler 不等于真正 beam search。第一阶段只接受 0/1；大于 1 返回“不支持”，直到实现独立 beam executor，禁止把 beam size 悄悄当作 top-k。 |
| 能力不支持 | tokenizer slow、多视觉编码器、预计算视觉 feature 等未必能由所选 llama.cpp 版本支持。setter 可以保存配置，但在加载/提交时必须返回“不支持”，不能无声忽略（头文件明确说对其他模型无效的 thinking mode 除外）。 |

### 推荐错误码

在不改变现有“0 成功、负值失败”约定的前提下增加公开枚举：

```c
enum inferlet_error_code {
    INFERLET_OK                  =  0,
    INFERLET_E_INVALID_ARGUMENT  = -1,
    INFERLET_E_INVALID_HANDLE    = -2,
    INFERLET_E_INVALID_STATE     = -3,
    INFERLET_E_IO                = -4,
    INFERLET_E_MODEL_LOAD        = -5,
    INFERLET_E_INFERENCE         = -6,
    INFERLET_E_UNSUPPORTED       = -7,
    INFERLET_E_CANCELLED         = -8,
    INFERLET_E_BUFFER_TOO_SMALL  = -9,
    INFERLET_E_OUT_OF_MEMORY     = -10,
    INFERLET_E_INTERNAL          = -11
};
```

## 3. 依赖与构建策略

### 3.1 llama.cpp 版本管理

- 以 Git submodule 放在 `third_party/llama.cpp`，固定到经过测试的 commit，不跟随上游浮动分支。
- 在 `doc/third_party.md` 记录 commit、编译选项、补丁和 MIT license 要求。
- wrapper 只调用该 commit 的公开 API；不要让业务代码依赖 `examples/server` 的私有实现。确需复用 `common`/`mtmd` 时，在单独 adapter 层隔离，因为这些 API 变动频繁。
- 每次升级 llama.cpp 单独提交，并跑完整 ABI、同步、异步、流式、LoRA 和多模态回归测试。

### 3.2 推荐 CMake 选项

```text
INFERLET_BUILD_TESTS=ON/OFF
INFERLET_BUILD_EXAMPLES=ON/OFF
INFERLET_ENABLE_CUDA=ON/OFF
INFERLET_ENABLE_VULKAN=ON/OFF
INFERLET_ENABLE_METAL=ON/OFF
INFERLET_ENABLE_MULTIMODAL=ON/OFF
INFERLET_ENABLE_SANITIZERS=ON/OFF
INFERLET_BUILD_FAT_ARCHIVE=ON/OFF
```

默认先交付 Linux CPU 构建。CUDA/Vulkan/Metal 分别作为独立构建矩阵，不要在首个里程碑中混在一起调试。

### 3.3 `libinferlet.a` 的关键事实

`target_link_libraries(Inferlet PRIVATE llama)` 不会把 llama/ggml 的对象自动装进 `libinferlet.a`；最终应用通常仍要显式链接 llama/ggml。若验收标准是调用方只看到一个项目静态库，则必须：

1. 先正常构建 wrapper、llama、ggml 的静态 archive；
2. 在打包阶段用平台对应工具合并 archive（Linux 可用 MRI `ar -M`，macOS 用 `libtool -static`）；
3. 对最终 archive 执行 `ranlib`，并用 `nm` 检查没有遗漏预期对象；
4. GPU runtime、`pthread`、`dl`、`m`、C++ runtime 等系统依赖仍无法可靠塞进 archive，应由导出的 CMake target 或 `.pc` 文件传递。

因此推荐的用户方式是 `target_link_libraries(app PRIVATE Inferlet::Inferlet)`；同时提供一条经过 CI 验证的裸编译器链接命令。

## 4. 推荐目录结构

```text
Inferlet/
├── CMakeLists.txt
├── cmake/
│   ├── InferletFatArchive.cmake
│   └── InferletConfig.cmake.in
├── include/
│   └── inferlet.h
├── src/
│   ├── api/inferlet_c_api.cpp
│   ├── core/runtime.{h,cpp}
│   ├── core/model_registry.{h,cpp}
│   ├── core/model_handle.{h,cpp}
│   ├── core/task_input.{h,cpp}
│   ├── core/task_output.{h,cpp}
│   ├── core/inference_task.{h,cpp}
│   ├── core/scheduler.{h,cpp}
│   ├── engine/llama_engine.{h,cpp}
│   ├── engine/prompt_builder.{h,cpp}
│   ├── engine/sampler.{h,cpp}
│   ├── engine/token_codec.{h,cpp}
│   ├── engine/beam_search.{h,cpp}
│   ├── multimodal/multimodal_backend.{h,cpp}
│   ├── multimodal/mtmd_backend.{h,cpp}
│   └── support/{error,logging,json_config,validation}.{h,cpp}
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── abi/
│   └── data/README.md
├── examples/{sync,async,stream,tokenizer}.cpp
├── doc/
└── third_party/llama.cpp/
```

所有导出符号集中在一个 C API translation unit 中。该层只做参数检查、异常屏蔽和内部对象转发，不在其中堆积推理逻辑。

## 5. 内部架构和对象模型

```text
C API
  -> 参数/句柄校验
  -> Runtime（init 引用计数、backend 生命周期、全局 registry）
  -> ModelHandle（调用线程、模型引用、任务集合）
  -> Scheduler（队列、worker、取消、完成通知）
  -> InferenceTask（每任务 context/KV/sampler/输出状态）
  -> LlamaEngine（llama.cpp 唯一适配层）
```

建议的核心对象：

- `Runtime`：进程级单例，维护 Init/inferlet_deinit 引用计数、`llama_backend_init/free`、worker 管理和模型 registry。首次 Init 初始化，最后一次 inferlet_deinit 才销毁。
- `ModelConfig`：`Parameter` 的值对象，所有字符串/数组自持有，并记录“用户是否显式设置该字段”。JSON 配置先加载，显式 setter 总是覆盖 JSON，与调用顺序无关。
- `SharedModel`：持有 `llama_model`、vocab、模型 metadata、LoRA adapters、设备配置和可选视觉 backend。由同路径的多个 `ModelHandle` 共享。
- `ModelHandle`：持有 `shared_ptr<SharedModel>`、创建线程 ID、任务表及关闭状态。它本身不直接保存某个任务的 KV cache。
- `TaskInput`：可变 builder；提交时转换为不可变 `TaskRequest`。字符串、tokens、chat、tools、choices、多模态字节都深拷贝。
- `InferenceTask`：状态机、取消标志、`llama_context`、sampler、输出累积、异常信息、mutex/CV。一个并发任务对应一个 context，先保证正确性，再优化 context pool/KV 复用。
- `TaskOutput`：只含拥有所有权的稳定内存。二维 `float **` 由一块连续数据加一组行指针组成，保证 getter 返回值到 Deallocate 前稳定。
- `StreamChunk`：保存本批新增 text/tokens/offsets/probability/logits/embedding。每次 `inferlet_get_stream_text` 把队列头移动到 task 的“当前 chunk”，各 stream getter 读取当前 chunk。

所有 `void *` 对象都应有内部类型标签、版本、状态和调试 magic。入口统一校验 null、类型和状态。注意：magic 无法让任意野指针变安全，因此仍要清楚记录“释放后不得再次传入”。

### 5.1 任务状态机

```text
Created -> Queued -> Running -> Completed -> Consumed
                    |     |
                    |     -> Failed -> Consumed
                    -> Cancelling -> Cancelled -> Consumed
```

- `inferlet_query_task` 在 Queued/Running/Cancelling 时返回 0 和空 output。
- Completed 时原子地交付 output 并回收 task。
- Failed/Cancelled 时返回对应负错误码，并回收 task。
- `inferlet_wait_task` 使用同一交付逻辑，只是通过 CV 等待终态。
- `inferlet_cancel_task` 是幂等的：设置取消标志；worker 在 prompt batch 之间和每个生成 token 之间检查。
- `inferlet_check_backend_active` 建议定义为该 handle 存在 loading、queued 或 running 工作时返回 1；只有终态但尚未消费的任务不算 active。

## 6. llama.cpp 映射方案

### 6.1 模型加载与上下文

- `inferlet_init/inferlet_deinit` 对应 llama backend 的一次性初始化/释放，以引用计数保护。
- `inferlet_load_from_file` 将 Parameter 转换成 model/context 配置，规范化路径，通过 registry 获取或创建 `SharedModel`。
- model 是共享且只读的；context/KV cache 是任务私有的。不要在并发任务间共享一个 `llama_context`。
- `DecoderMaxTokenCnt` 映射为 prompt 分块和 decode batch/ubatch 上限，不应误当作 context window。
- prompt tokens + max output 超过 context 大小时，第一版明确返回错误；以后若增加截断/KV shift 策略，应由配置显式选择。
- `TokenizerOnly` 映射为 vocab-only/model 的最低资源加载模式，并禁止 Run，只允许 encode/decode。

### 6.2 Prompt 构建

建立 `PromptBuilder`，输出统一的 `PreparedPrompt { tokens, token_text, multimodal_segments }`：

- PromptText：按模型 tokenizer 编码，是否加入 BOS 由模型 metadata/chat template 决定。
- PromptTokens：原样校验并使用，不再次编码。
- ChatHistory：转成 role/message 序列，通过 pinned llama.cpp 的 chat template API 渲染；对没有模板的模型返回明确错误，或使用 run config 指定模板。
- ThinkingMode：仅在确认支持的模型模板中切换；其他模型按头文件约定无效果。
- Tools：先解析每项 JSON，再交给模型特定 prompt adapter；不能只是把原始 JSON 任意拼接到 prompt。
- GuidedChoices：转换成严格转义的 grammar/trie 约束，确保只能生成候选项之一，然后接收 EOS。
- MultiModal：由独立 backend 生成 text token 与 embedding batch 的交错序列，不让多模态逻辑侵入普通文本路径。

### 6.3 生成循环

同步和异步/流式必须共用同一个 `generate(TaskRequest, Sink)` 内核，只更换结果 sink：

1. 校验 request 和模型能力；创建 context 与每任务 sampler。
2. 构造 prompt，并按 batch 分块 decode。
3. 从最后位置取得原始 logits；若要求 verbose，立即深拷贝。
4. 计算 type=1 概率，应用 repetition penalty、temperature、top-k、top-p 等 sampler，再计算 type=2 概率。
5. sample token；遇到 EOS/模型终止 token 时结束，且按冻结后的契约决定是否输出该 token。
6. 将 token decode 进 context；若要求 embedding，在 llama.cpp buffer 失效前深拷贝。
7. token 转 piece，正确处理 special-token 策略和 UTF-8 分段，记录字节 offset。
8. 写入 collecting sink 或 streaming sink；检查取消、max tokens 和错误。
9. 正常结束时提交 output/EOF；任何异常转换成稳定错误码。

不要对每个 token 重建 sampler；sampler 状态属于 task。随机种子暂未暴露在头文件中，应在 run config 中定义，集成测试固定 seed。

### 6.4 Sampler 顺序

必须固定顺序并写测试，建议为：

```text
raw logits
-> repetition penalty
-> guided grammar/choices
-> temperature（temperature <= 0 时 greedy）
-> top-k
-> top-p
-> categorical distribution
```

grammar 必须在 top-k/top-p 截断前屏蔽非法 token，否则合法候选可能先被截断而导致候选集为空。

具体顺序一旦发布会影响可复现性，不应随 llama.cpp 升级而无意改变。`top_k=0` 使用项目默认值；其他默认值全部集中定义在 `TaskInput` 构造函数和公开文档中。

### 6.5 LoRA、设备与多 backend

- Parameter 阶段按 ID 加载/登记 LoRA 文件，`lora_id=0` 表示不应用。任务创建 context 后按 `SetLowRankAdapterId` 选择 adapter；未配置的 ID 返回错误。
- `DeviceId` 映射主设备；`DeviceList` 映射 llama.cpp backend device/tensor split。CPU-only 构建只接受 CPU 的有效配置。
- `BackendCnt` 定义为可同时执行任务的 engine/context worker 数，而不是复制 model。先验证头文件要求的 `device_cnt / backend_cnt` 为 2 的幂；非法配置因 setter 无返回值，应给 Parameter 记 invalid 状态并在 Load 时失败。
- context pool 是后续性能优化。首版每任务创建/销毁 context，确认正确后再以“模型、context size、LoRA、设备”作为 pool key。

### 6.6 多模态边界

多模态不要纳入文本 MVP。先定义 `MultimodalBackend` 能力接口，再用 pinned llama.cpp 对应的 `mtmd`/模型专用组件实现。

必须验证：

- `type`、data、size、grid、data_type、frame_num、packing_num 的组合和乘法溢出；
- 同一 task 的 feature data type 一致；
- raw uint8 `[t,c,h,w]` 的模型特定 `c/patch_size/temporal_patch_size`；
- feature 的 `n_embd` 与 decoder 匹配；
- role 拼写兼容策略（头文件注释中 `assisstant` 有拼写错误，正式契约应只接受或规范成 `assistant`）；
- `cache_key` 至少以 model identity、visual encoder ID、shape、dtype 共同组成命名空间；相同 key 但 metadata 不同必须报错；
- `cache_times` 用原子计数管理，并设置总内存上限和 LRU 兜底，防止恶意 key 占满内存；
- 不是所有 llama.cpp 模型都支持外部预计算 feature 注入，不支持时返回 `INFERLET_E_UNSUPPORTED`。

## 7. 公开 API 实现清单

| API 族 | 内部实现 | 完成标准 |
| --- | --- | --- |
| Init/inferlet_deinit | `Runtime` 引用计数和 backend 生命周期 | 多次、交错调用无泄漏；最后一次 inferlet_deinit 等待后台安全退出 |
| Parameter Allocate/Deallocate | `ModelConfig` builder | 默认值、深拷贝、非法配置延迟错误均有单测 |
| Parameter setters | model/context/device/LoRA/vision/run-config 字段 | 每项都有默认值、校验规则和 llama.cpp 映射文档 |
| Load/Unload | registry + `SharedModel` + `ModelHandle` | 同模型复用、不同模型隔离、加载失败无半初始化对象 |
| TaskInput Allocate/Deallocate | `TaskInput` builder | 所有 setter 深拷贝，重复 setter 不泄漏，OOM 可恢复 |
| 内容 setters | PromptBuilder 输入 variant | 互斥和 last-wins 语义一致；空输入规则明确 |
| 采样 setters | `SamplingConfig` | 边界值拒绝；默认值和 sampler 顺序固定 |
| Run | collecting sink | 文本、tokens、offset、概率完全对齐 |
| inferlet_run_async/Query/Wait/Cancel | Scheduler + task 状态机 | 每个终态只交付/回收一次，无竞态或 double free |
| inferlet_run_stream/GetStream* | streaming sink + chunk queue | block/non-block 模式、空 chunk、EOF、取消语义全部覆盖 |
| TaskOutput getters | `TaskOutput` 稳定存储 | Deallocate 前所有返回指针稳定，未启用数据返回 NULL |
| Verbose getters | 推理循环即时拷贝 | shape 正确，最后一个 token 无 off-by-one，内存上限可控 |
| Tokenizer Encode/Decode | `TokenCodec` | 两遍容量查询、UTF-8、special token、截断行为有测试 |
| GuidedChoices/Tools/Thinking | model-specific prompt/sampler adapter | 支持矩阵明确；不支持绝不静默降级 |
| LoRA | adapter registry + per-context apply | 多 ID 切换、缺失 ID、并发 context 均正确 |
| MultiModal | `MultimodalBackend` | text/image/video 组合、feature/raw image、缓存生命周期通过模型测试 |

## 8. 分阶段实施计划

### M0：契约、依赖和工程骨架

- 完成第 2 节所有决策，修正头文件 C 兼容性和注释，不改变已有函数 ABI。
- 固定 llama.cpp commit 和 license，建立顶层 CMake、静态库、安装规则、C/C++ smoke test。
- 实现统一的错误转换、日志、句柄校验和空壳导出函数；未实现函数返回 `INFERLET_E_UNSUPPORTED`，绝不能返回假成功。
- 建立 `nm` 导出符号基线，核对头文件的 64 个函数全部存在。

验收：一个外部最小项目能只包含安装后的 `inferlet.h`，通过 CMake target 链接 `libinferlet.a`；C 与 C++ 头文件编译测试均通过。

### M1：Runtime、模型与 tokenizer

- 实现 Init/inferlet_deinit、Parameter 基础字段、同步 Load/Unload、model registry。
- 实现 tokenizer-only 模式和 Encode/Decode，包括容量不足、special token 和 UTF-8 测试。
- 实现 run config 的 versioned JSON schema；建议字段覆盖日志、seed、threads、context size、batch size、默认 sampler 和内存上限。

验收：可加载一个小型 GGUF，encode/decode 与同 commit 的 llama.cpp 基准程序结果一致；循环加载卸载无泄漏。

### M2：文本同步推理 MVP

- 实现 PromptText/PromptTokens、采样 setter、max output、special token mode。
- 实现单任务 context、prompt decode、greedy/top-k/top-p/temperature/repetition sampling 和 collecting sink。
- 实现 TaskOutput 文本、tokens、offset、probability getter。
- 此阶段 `beam_size > 1`、tools、guided、多模态和 LoRA 明确返回 unsupported。

验收：固定模型、prompt、seed 的输出可复现；tokens 重新 decode 后与 result text 契约一致；ASan/UBSan 无错误。

### M3：Chat 与完整普通输出

- 实现普通 ChatHistory、模型 chat template、thinking mode 能力判断。
- 实现 input-token verbose 信息。
- 实现 logits/embedding 捕获和二维稳定存储，加入单任务 verbose 内存上限；超过上限应在提交前或运行中明确失败。
- 检查 logits、embedding、token 的时间步对齐，尤其是首个和最后一个生成 token。

验收：至少覆盖一个有 chat template 的模型和一个无模板模型；用直接 llama.cpp 调用交叉验证每个时间步的 logits。

### M4：异步调度与取消

- 实现 Scheduler、worker、状态机、inferlet_run_async、inferlet_query_task、inferlet_wait_task、inferlet_cancel_task、inferlet_check_backend_active。
- 每任务独立 context；按 `BackendCnt` 限制并发，其余任务排队。
- 覆盖排队时取消、prompt decode 中取消、生成中取消、完成与取消竞争、Unload 与任务竞争。

验收：压力测试反复提交/查询/等待/取消，无死锁、double free、遗漏唤醒；TSan 构建通过。验证同一 model 的多个 handle 可并发且 model 数据只加载一份。

### M5：流式推理

- 复用 M2 的生成内核，增加 streaming sink 和有界 chunk queue（提供背压，防止调用方不读取导致无限增长）。
- 实现阻塞/非阻塞 `inferlet_get_stream_text`，随后各 getter 读取完全相同的 current chunk。
- 聚合不完整 UTF-8 字节，避免对调用方返回非法 UTF-8；不可见 special token 不应让阻塞模式无限返回空串。
- 实现 EOF 自动回收和提前 Cancel 回收路径。

验收：拼接所有 stream text 与同配置同步结果一致；拼接 stream tokens/概率也一致；慢消费者、取消、空 piece 和 EOF 均无泄漏。

### M6：高级文本能力

- GuidedChoices：grammar/trie、JSON/GBNF 转义、与 sampler 顺序的组合测试。
- Tools：JSON 校验、model template adapter、支持模型矩阵。
- LoRA：加载、ID registry、per-task 切换和并发。
- 真正 beam search 单独实现和评审；需要 beam score、KV sequence copy、终止条件、length penalty 和最终候选选择。在完成前只允许 beam=1。
- 最后再实现异步加载；必须解决后台加载失败的可观察性。

验收：每种高级能力既有成功模型测试，也有“不支持模型”的明确失败测试；不允许行为降级但仍返回 0。

### M7：设备、多 backend 和性能

- 映射 DeviceId/DeviceList，增加 CPU、CUDA（及需要的平台）CI 构建。
- 实现 context pool、批量 prompt decode、合理的线程配置和性能指标。
- 性能优化前后执行完全相同的正确性回归；不得为了复用 context 泄漏 KV 或 sampler 状态。

验收：记录首 token 延迟、tokens/s、峰值 RSS/显存、并发吞吐和取消延迟；与同配置 llama.cpp CLI/benchmark 比较并解释差异。

### M8：多模态

- 先实现 `MultiModal` 的 text + 单图路径，再实现 MultiModal ChatHistory、多图、视频、外部 feature 和缓存。
- 将 vision 参数映射到所选 backend；未实现的参数在加载/提交时失败。
- 添加每个支持模型族的 golden prompt/shape 测试，不能假定所有 VL 模型有相同 placeholder 或 patch merge 规则。

验收：至少选择一个明确受 pinned llama.cpp 支持的 VL GGUF 完成端到端测试；raw uint8 和预计算 feature 分别验证。缓存命中输出与未缓存一致，引用次数和淘汰无泄漏。

### M9：发布与兼容性

- 生成 fat archive、CMake package、pkg-config、examples、API 文档和支持矩阵。
- CI 覆盖 Debug/Release、GCC/Clang、CPU/GPU、ASan/UBSan/TSan（按可用环境拆分）。
- 固定 ABI symbol list；发布后只允许追加函数，不改变结构体已有字段顺序、函数签名或语义。
- 验证 clean machine 安装和外部消费项目，不允许测试意外从源码树找到依赖。

验收：安装包解压后，示例项目可以独立编译运行；`nm`/ABI 检查、license 清单和完整测试报告通过。

## 9. 测试计划

### 9.1 单元测试

- Parameter/TaskInput 默认值、深拷贝、重复设置、空值、边界值和整数溢出。
- JSON 配置优先级、未知字段、版本不兼容。
- prompt builder、chat template、guided choice 转义。
- token piece 拼接、UTF-8 分片、offset。
- task 状态机的所有合法/非法迁移。
- stream queue 背压、EOF 和 block mode。
- 二维 logits/embedding 行指针与连续存储。

可实现仅测试使用的 deterministic fake engine，以毫秒级覆盖异步竞态和错误注入；真实 llama.cpp 集成测试不能被 fake 测试替代。

### 9.2 集成测试

- 小型文本 GGUF：tokenizer、greedy、sample、chat、同步、异步、流式。
- LoRA fixture：无 adapter、合法 ID、非法 ID。
- VL GGUF：图像、视频/多帧、feature、缓存。
- 模型文件不存在、损坏、OOM/上下文超限、任务取消、Unload 竞争。
- 同步结果与所有流 chunk 拼接结果一致。
- 同一路径两个 handle 共享 model；不同任务 KV/采样状态互不污染。

模型通常较大，不要直接提交到 Git；用脚本按固定 SHA256 获取，离线 CI 则配置本地 fixture 路径。

### 9.3 ABI 和打包测试

- 一个 `.c` 与一个 `.cpp` 消费者只包含公开安装目录。
- 比对头文件声明和 `nm -g --defined-only libinferlet.a` 的导出列表。
- 检查 public header 不泄漏 C++/llama.cpp 类型。
- clean build 验证 fat archive 中包含 llama/ggml 所需对象。
- 对安装后的 `Inferlet::Inferlet` 做一次真实链接，验证系统/GPU传递依赖。

### 9.4 质量门槛

- 单元和集成测试全部通过。
- ASan/UBSan 无问题；并发阶段 TSan 无数据竞争。
- clang-format/clang-tidy 只对本项目源码运行，不批量改写 third_party。
- 所有 C API 入口有 null/范围/状态校验和异常边界。
- 重复 Init/Load/Run/Unload/inferlet_deinit 压力循环后，无任务、context、model、adapter 和缓存泄漏。

## 10. 风险与控制措施

| 风险 | 控制措施 |
| --- | --- |
| llama.cpp API 快速变化 | 固定 commit；所有调用收敛到 `LlamaEngine`/multimodal adapter；升级单独回归 |
| 头文件语义不完整 | M0 先冻结契约；测试以契约而不是当前偶然行为为准 |
| 静态库没有真正包含依赖 | fat archive 专门构建与 clean consumer 链接测试 |
| 同 model 多任务误共享 context/KV | model 只读共享，每任务 context；后续 pool 在归还时完整清理 |
| stream 自动释放导致 use-after-free | 明确 EOF 后禁用句柄；统一终态回收；提前取消路径必须可回收 |
| verbose logits 占用巨大内存 | 默认关闭；提交前估算 `max_tokens * n_vocab * sizeof(float)`；配置硬上限 |
| 多模态跨模型差异大 | backend 和模型能力表；逐模型 family 实现，unsupported 明确报错 |
| setter 为 void 无法立即报告非法 Parameter | Parameter 保存 invalid reason，Load 时失败并输出详细日志；后续 ABI 可追加返回 int 的新接口 |
| 随机结果难回归 | run config 支持 seed；测试使用 greedy 或固定 seed，并锁定 sampler 顺序 |
| inferlet_deinit/Unload 与任务竞态 | close -> cancel -> join -> destroy 的固定销毁顺序；TSan 和压力测试覆盖 |

## 11. 建议的首个可交付版本

首版定义为 M0 至 M3，支持：

- Linux CPU、单设备；
- GGUF 文本模型同步加载；
- prompt text、prompt tokens、普通 chat；
- greedy、top-k、top-p、temperature、repetition penalty；
- 同步推理和完整 TaskOutput；
- tokenizer encode/decode；
- 可选 input tokens、logits、embedding verbose 输出。

异步、流式、LoRA、beam、tools、guided choices 和多模态在对应里程碑完成前均返回明确的 unsupported 错误。这样可以尽早获得一个行为可靠、可测试、可供其他项目链接的 `libinferlet.a`，同时不为尚未实现的高级 API 制造“返回成功但实际无效”的兼容性负担。

## 12. 每个里程碑的提交顺序

每个里程碑建议按以下粒度提交，便于 review 和回退：

1. 契约/测试先行：新增失败测试和文档。
2. 内部类型及 llama adapter，不导出新行为。
3. C API 接线和异常边界。
4. 集成模型测试与内存/并发检查。
5. 示例、安装与发布说明。

只有当前里程碑验收项全部通过后再进入下一阶段。尤其是 M4/M5 的并发和生命周期问题，必须在引入多模态与 context pool 之前解决。

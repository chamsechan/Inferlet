# Inferlet 详细设计与逐步实施手册

## 0. 文档定位

本文是 [`inferlet_implementation_plan.md`](./inferlet_implementation_plan.md) 的落地设计。前者回答“按什么阶段做”，本文回答：

- 每个公开函数应具有什么行为；
- 内部对象有哪些字段、谁拥有它们、何时销毁；
- llama.cpp 应只在哪一层被调用；
- 同步、异步、流式推理如何共用一套生成内核；
- 每一步应创建哪些文件、实现什么、运行什么测试；
- 做到什么程度才能进入下一步。

本文中出现的 llama.cpp 函数名用于说明映射关系。llama.cpp 上游变化较快，实际编码前必须固定一个 commit，并以该 commit 的头文件为唯一准绳。官方 master 的 [`llama.h`](https://github.com/ggml-org/llama.cpp/blob/master/include/llama.h)、[`examples/simple`](https://github.com/ggml-org/llama.cpp/blob/master/examples/simple/simple.cpp) 和 [`tools/batched-bench`](https://github.com/ggml-org/llama.cpp/blob/master/tools/batched-bench/batched-bench.cpp) 只能作为选择 commit 时的参考，不能在构建中跟踪 master。

---

## 1. 设计基线

### 1.1 最终目标

安装后的调用方只接触：

```text
<prefix>/include/inferlet.h
<prefix>/lib/libinferlet.a
<prefix>/lib/cmake/Inferlet/InferletConfig.cmake
<prefix>/lib/pkgconfig/inferlet.pc
```

调用方能够使用头文件中的全部 API；不支持某模型或构建后端的功能必须返回稳定错误，不能崩溃、假成功或静默忽略。

### 1.2 第一条基线

先以以下环境打通，随后再增加平台矩阵：

- Linux x86_64；
- GCC 和 Clang；
- C++17 内部实现；
- llama.cpp CPU backend；
- GGUF 文本生成模型；
- 静态构建；
- Debug、Release、ASan/UBSan 和 TSan 配置。

CUDA/Vulkan/Metal、LoRA、beam search、多模态分别放在基础文本链路稳定以后接入。平台扩展不能改变已经发布的 C ABI。

### 1.3 明确不做的事情

- 不在公共头文件暴露 llama.cpp、ggml 或 C++ STL 类型。
- 不负责联网下载模型。
- 不把 llama.cpp server 的 HTTP 层带入库中。
- 不在第一版实现跨请求 prefix cache、continuous batching 或 context shift。
- 不将 `beam_size` 偷换为 top-k。
- 不为不支持的多模态模型猜测 prompt/patch 规则。

### 1.4 完成定义

项目目标只有同时满足下列条件才算完成：

1. 头文件声明的 64 个函数都有导出实现。
2. 每个函数符合本文定义的参数、返回值、生命周期和线程语义。
3. 文本同步、异步、流式、tokenizer 的真实 GGUF 集成测试通过。
4. LoRA、guided choices、tools、beam 和多模态要么通过支持模型测试，要么对不支持场景返回 `INFERLET_E_UNSUPPORTED`。
5. `libinferlet.a` 通过干净目录中的外部 consumer 链接测试。
6. ASan/UBSan 无错误；并发路径通过 TSan 和压力测试。
7. Init/inferlet_deinit、Load/Unload、task 和 stream 的重复循环无资源泄漏。

---

## 2. 公共 ABI 的确定设计

### 2.1 头文件先行修正

开始写实现前先修改 `include/inferlet.h`：

1. 将 include guard 改为 `INFERLET_H_INCLUDED`。
2. 将五个公开结构改成 C/C++ 都能使用的 typedef 形式，例如：

   ```c
   typedef struct inferlet_chat_history {
       const char *system;
       const char *query;
       const char *response;
   } inferlet_chat_history;
   ```

3. `INFERLET_API` 增加 Windows/共享库分支；静态库分支为空。
4. 增加公开错误码枚举，但保持已有函数签名不变。
5. 建议追加 `const char *inferlet_get_last_error(void)`；若现阶段绝对不能增加函数，则所有详细错误写入 stderr/回调日志，原 API 只返回错误码。
6. 修正文档中的 `assisstant` 为 `assistant`，以及 logits/embedding 注释的复制错误。
7. 所有数组入参能保持 ABI 的地方改为 `const`，例如 prompt tokens 和 device list。若要求与既有二进制严格兼容，可以只在内部当作只读，不立即改签名。

头文件应在 CI 中分别通过：

```sh
cc  -std=c11   -fsyntax-only tests/abi/header_c.c
c++ -std=c++17 -fsyntax-only tests/abi/header_cpp.cpp
```

### 2.2 通用调用规则

- `Allocate` 返回的对象只能交给同族函数。
- 所有 `Deallocate(NULL)` 和 `Unload(NULL)` 定义为无操作。
- 其他函数遇到必要指针为 NULL 时返回 `INFERLET_E_INVALID_ARGUMENT`；返回指针的 getter 返回 NULL，并把可写 length 设为 0。
- setter 深拷贝全部输入。setter 成功返回后，调用方可立刻释放原内存。
- `Run*` 提交时再深拷贝/移动成不可变 `TaskRequest`，因此提交后调用方可修改、复用或释放 `task_in`。
- 所有跨 ABI 的函数体放在 `try/catch (...)` 中；`std::bad_alloc` 转换为 `INFERLET_E_OUT_OF_MEMORY`，其他异常转换为 `INFERLET_E_INTERNAL`。
- 不允许记录调用方传入的临时 `const char *`、数组或 `void *feature` 裸指针。
- 返回的内部指针只在各自文档规定的有效期内有效，调用方不得释放。

### 2.3 错误码

```c
typedef enum inferlet_error_code {
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
} inferlet_error_code;
```

内部 `Status` 同时保存 code 和 message。C API 每次失败时写线程局部 `last_error`；每次成功时清空，避免调用方读到上一次错误。

### 2.4 句柄和生命周期

| 公开指针 | 内部类型 | 创建 | 唯一销毁点 |
| --- | --- | --- | --- |
| `param` | `Parameter` | `inferlet_parameter_allocate` | `inferlet_parameter_deallocate` |
| `handle` | `ModelHandle` | `inferlet_load_from_file` | `inferlet_unload` |
| `task_in` | `TaskInput` | `inferlet_task_input_allocate` | `inferlet_task_input_deallocate` |
| `task` | `TaskTicket` | `inferlet_run_async` | 首次完成交付的 Query/Wait；取消后也由 Query/Wait 回收 |
| `task_stream` | `StreamTicket` | `inferlet_run_stream` | inferlet_get_stream_text 返回 NULL 时；或 inferlet_cancel_task 明确回收 |
| `task_out` | `TaskOutput` | Run/Query/Wait | `inferlet_task_output_deallocate` |

禁止同一 ticket 被两个线程同时 Query/Wait。头文件已经声明同一 model handle 不是线程安全的，调用层不为这种误用提供正常业务语义；Debug 构建应检测创建线程。

### 2.5 数据默认值和校验

#### TaskInput 默认值

| 字段 | 默认值 | 校验 |
| --- | --- | --- |
| beam size | 1 | `>=1`；大于 1 走 beam executor |
| top-k | 40；setter 传 0 恢复默认 | `>=0` |
| top-p | 0.95 | `(0,1]` |
| temperature | 0.8 | `>=0`；0 表示 greedy |
| repetition penalty | 1.0 | `>0` |
| repeat last n | 64，只能由 JSON 配置 | `>=0` |
| max output tokens | 256 | `>0` |
| probability type | 0 | 0、1、2 |
| verbose flag | 0 | 只接受已定义 bit 或 `INFERLET_VERBOSE_F_ALL` |
| stream block | 0 | 0、1 |
| thinking | 1 | 0、1 |
| keep special token | 0 | 0、1 |
| LoRA ID | 0 | `>=0`；0 表示无 LoRA |
| visual encoder | 空 | 空表示默认 |

这些值是 Inferlet v1 契约，不跟随 llama.cpp 默认值变化。若产品需要不同默认值，应在首次发布前统一修改本文、头文件注释和测试。

#### Parameter 默认值

| 字段 | 默认值 | 含义 |
| --- | --- | --- |
| tokenizer only | 0 | 加载完整模型 |
| tokenizer slow | 0 | 仅支持 backend 有对应实现时使用 |
| vision limits | 0 | 使用模型/backend 默认 |
| vision transform skipped | 0 | 输入原始数据时运行视觉编码 |
| decoder max token count | 512 | 映射 decode `n_batch`/`n_ubatch` 上限 |
| device ID | -1 | 自动选择；CPU-only 构建为 CPU |
| device list | 空 | 自动设备列表 |
| backend count | 1 | 同一 shared model 最大同时执行任务数 |
| load sync | 0 | 按原头文件说明，默认异步加载 |
| run config | 空 | 使用库默认配置 |

Parameter setter 为 void，无法立刻返回错误。每次 setter 设置 `parameter.valid=false` 和 `invalid_reason`；`inferlet_load_from_file` 检测后返回 NULL 并设置 last error。

### 2.6 内容 setter 的互斥规则

`TaskInput` 使用 variant 保存一种主要输入：

```text
monostate | PromptText | PromptTokens | Chat | LegacyMultiModal | MultiModalChat
```

- 任一内容 setter 成功时替换前一种内容，即 last-wins。
- 空字符串是合法 prompt；NULL prompt 非法。
- tokens 长度可为 0，此时 tokens 可为 NULL；长度大于 0 时 tokens 不能为 NULL。
- 数组 length 可为 0，此时数组可为 NULL；length 大于 0 时数组不能为 NULL。
- 同一 `TaskInput` 的采样、LoRA、tools 等配置不因内容 setter 替换而清空。

### 2.7 每组 API 的精确行为

#### 全局与模型

| API | 前置条件 | 成功行为 | 失败/特殊行为 |
| --- | --- | --- | --- |
| `inferlet_init` | 无 | 引用计数 +1；首次调用初始化 llama backend 和 runtime | void；异常记录日志，不抛出 |
| `inferlet_deinit` | 应先 Unload | 引用计数 -1；计数为 0 且无活对象后关闭 runtime | 多余 inferlet_deinit 只告警；不产生负计数 |
| `inferlet_load_from_file` | 已 Init；路径存在；param 可 NULL | 创建 ModelHandle；同步模式加载完成后返回，异步模式创建 loading handle 后返回 | 参数/同步加载失败返回 NULL；异步失败由 Check/Run 报告 |
| `inferlet_unload` | handle 可 NULL | close、取消所属任务、join，再释放 handle 引用 | NULL 无操作；错误只记录日志 |
| `inferlet_check_backend_active` | 合法 handle 和 is_active | loading/queued/running 时写 1，否则 0 | 异步加载失败返回 `MODEL_LOAD` 并写 0 |

#### 推理执行

| API | 精确语义 |
| --- | --- |
| `inferlet_run` | 创建 TaskRequest，在当前调用内等待模型 ready 和推理完成；成功生成一个独立 TaskOutput。失败时 `*task_out=NULL`。 |
| `inferlet_run_async` | 创建普通 task，入队后立即返回。`*task` 只在返回 0 时有效。模型仍在异步加载时任务保持 queued。 |
| `inferlet_run_stream` | 创建 stream task，入队后立即返回；chunk queue 有界。 |
| `inferlet_wait_task` | 等待普通 task 终态；完成时将 output 所有权交给调用方并销毁 ticket。失败/取消也销毁 ticket。 |
| `inferlet_query_task` | 未完成返回 0、`*task_out=NULL`；完成则交付 output 并销毁 ticket；失败/取消返回负值并销毁 ticket。 |
| `inferlet_cancel_task` | 普通 task：只设置取消，请继续 Query/Wait 回收。stream task：设置取消、等待 worker 退出、清空 chunk 并直接回收 ticket。重复取消正在取消的普通 task 返回 0。 |

`inferlet_cancel_task` 对 stream 的“直接回收”必须写入头文件，因为它与普通 task 不同。若不接受这种差异，应新增统一的 `Task_Deallocate/Stream_Deallocate`，不要留下无法回收的 stream。

#### TaskInput

| API | 实现要求 |
| --- | --- |
| Allocate/Deallocate | 创建默认 builder；NULL deallocate 无操作 |
| SetPromptText | 深拷贝到 `PromptText` |
| SetPromptTokens | 深拷贝 `int[length]`，逐 token 检查能否转换成 llama token 类型 |
| SetChatHistory | 深拷贝每个字符串；展开规则见第 6.1 节 |
| SetMultiModal | 深拷贝 prompt 和 feature 字节；做 size 上限与溢出校验 |
| SetMultiModal_ChatHistory | 递归深拷贝 role/content/ext/data；校验所有长度和 dtype |
| SetLowRankAdapterId | 保存 ID；真正检查是否已加载发生在提交时 |
| SetBeamSize/TopK/TopP/Temperature/RepetitionPenalty/MaxOutputTokens | 立即范围校验；失败时保持旧值不变 |
| SetProbabilityType | 只接受 0/1/2 |
| SetVerboseDataFlag | 保存规范化掩码；ALL 转为所有已知 bit |
| SetStreamBlockMode | 只影响 inferlet_run_stream；同步/普通异步忽略 |
| SetThinkingMode | 保存布尔值；不支持的非 Qwen 模型按头文件约定忽略并记录 debug 日志 |
| SetSpecialTokenMode | 只影响 token-to-piece 生成文本，不改变 result token IDs |
| SetGuidedChoices | 深拷贝非空 choices；num>0；拒绝重复和非法 UTF-8 |
| SetTools | 每项必须是合法 JSON object；深拷贝规范化 JSON |
| SetVisualEncoder | 深拷贝 ID；空字符串选择默认视觉编码器 |

#### TaskOutput

| Getter | 未启用/失败 | 返回内存有效期 |
| --- | --- | --- |
| ResultText | 空结果返回 `""`，非法 output 返回 NULL | Deallocate 前 |
| ResultTokens | 总是可取；空结果返回 NULL、length=0 | Deallocate 前 |
| TokenOffsets | 与 tokens 等长；空结果 NULL | Deallocate 前 |
| TokenProbability | probability type=0 时 NULL，否则与 tokens 等长 | Deallocate 前 |
| InputTokens | verbose bit 未开时 NULL、length=0、token_text=NULL | Deallocate 前 |
| OutputLogits | verbose bit 未开时 NULL；否则 `[length][n_vocab]` | Deallocate 前 |
| OutputEmbedding | verbose bit 未开时 NULL；否则 `[length][n_embd]` | Deallocate 前 |

`inferlet_task_output_deallocate(NULL)` 无操作。二维数据在 finalize 后建立行指针，之后禁止改变底层 vector 容量。

#### Stream getters

调用顺序固定为：

```text
inferlet_get_stream_text -> 可选调用本 chunk 的其他 getter -> inferlet_get_stream_text -> ... -> NULL
```

| API | 行为 |
| --- | --- |
| `inferlet_get_stream_text` | 非阻塞：无 chunk 且未结束返回 `""`；阻塞：等到有可见 chunk 或终态；有 chunk 时切换 current chunk；所有 chunk 耗尽且终态时返回 NULL 并销毁 ticket |
| `inferlet_get_stream_tokens` | 返回 current chunk tokens，并写 length；尚未取得 chunk 时 NULL/0 |
| `inferlet_get_stream_token_offsets` | 返回相对于 current chunk text 的 UTF-8 字节偏移 |
| `inferlet_get_stream_token_probability` | type=0 返回 NULL，否则与 current chunk tokens 等长 |
| `inferlet_get_stream_output_logits` | 返回 current chunk 二维数据和 shape |
| `inferlet_get_stream_output_embedding` | 返回 current chunk 二维数据和 shape |
| `inferlet_get_stream_input_tokens` | 只在第一次有输出的 chunk 返回输入 token 信息，之后返回 NULL；必须在头文件注明“一次性” |

current chunk 的所有指针在下一次 `inferlet_get_stream_text` 或 ticket 销毁前有效。

#### Parameter 与 tokenizer

| API | 设计 |
| --- | --- |
| Parameter Allocate/Deallocate | 创建/销毁 ModelConfig builder |
| TokenizerOnly | 映射 model params 的 vocab-only 能力；Run* 返回 unsupported |
| TokenizerSlow | backend 不具备对应 tokenizer 时，将 Parameter 标 invalid 或 Load 返回 unsupported |
| VisionMaxPatchCnt/GridSize/TransSkipped | 只交给 multimodal backend；纯文本模型忽略 0，非默认值返回 unsupported |
| DecoderMaxTokenCnt | 校验 >0，映射 context batch/ubatch 上限 |
| DeviceId/DeviceList | 规范化设备选择；两者同时设置时最后一次调用清除另一种配置 |
| BackendCnt | 校验 >0，以及 device/backend 关系；控制 shared model semaphore |
| LowRankAdapter | 深拷贝 ID->文件；ID>0、路径非空、ID 不重复覆盖时释放旧值 |
| LoadFromFileSync | 只接受 0/1；无效值使 Parameter invalid |
| RunConfigFile | 深拷贝路径；Load 时读取和验证 |
| inferlet_tokenizer_encode | 两遍式容量语义；`real_token_length` 始终返回完整所需 token 数 |
| inferlet_tokenizer_decode | 逐 token 转 piece 并拼接；`real_text_length` 不含 `\0`，容量包含 `\0` |

Tokenizer 容量不足返回 `INFERLET_E_BUFFER_TOO_SMALL`，同时写入能容纳的完整前缀和末尾 `\0`。不要截断在 UTF-8 code point 中间。

DeviceList 的具体规则：`device_cnt < 0` 使 Parameter invalid；`device_cnt == 0` 表示自动选择；`device_cnt > 0 && device_list == NULL` 按头文件约定生成 `[0, 1, ..., device_cnt-1]`；非空列表中的 ID 必须非负且不重复。BackendCnt 必须大于 0；已知 device count 时，先检查 `device_cnt % backend_cnt == 0`，再检查商是否为 2 的幂。

---

## 3. 内部数据结构

以下是设计形状，不要求逐字照搬命名，但字段所有权和并发职责不能改变。

### 3.1 基础类型

```cpp
enum class ObjectKind : uint32_t {
    Parameter, ModelHandle, TaskInput, TaskTicket, StreamTicket, TaskOutput
};

struct ObjectHeader {
    uint64_t magic;
    ObjectKind kind;
    uint32_t abi_version;
};

struct Status {
    int code = INFERLET_OK;
    std::string message;
    explicit operator bool() const { return code == INFERLET_OK; }
};
```

实现一个全局 `LiveObjectRegistry`。API 入口先按地址查询 registry，再读取 ObjectHeader，避免直接解引用一个明显已释放的地址。地址被 allocator 重用造成的陈旧句柄仍无法完全识别，因此释放后再次调用仍定义为调用方错误。

### 3.2 Parameter

```cpp
struct ModelConfig {
    bool tokenizer_only = false;
    bool tokenizer_slow = false;
    int vision_max_patch_cnt = 0;
    int vision_max_grid_size = 0;
    bool vision_transform_skipped = false;
    int decoder_max_token_count = 512;

    std::variant<AutoDevice, SingleDevice, DeviceList> devices;
    int backend_count = 1;
    std::map<int, std::string> lora_files;
    bool load_sync = false;
    std::string run_config_file;

    std::set<FieldId> explicitly_set;
    bool valid = true;
    std::string invalid_reason;
};

struct Parameter {
    ObjectHeader header;
    ModelConfig config;
};
```

JSON 读取时先构造 `json_config`，随后以 `explicitly_set` 中的 setter 值覆盖。这样配置优先级不受 setter 调用先后影响。

### 3.3 TaskInput 与不可变请求

```cpp
struct SamplingConfig {
    int beam_size = 1;
    int top_k = 40;
    float top_p = 0.95f;
    float temperature = 0.8f;
    float repetition_penalty = 1.0f;
    int repeat_last_n = 64;
    int max_output_tokens = 256;
    int probability_type = 0;
    uint64_t seed = 0; // 0 表示由 runtime 生成；测试固定非零值
};

struct OutputOptions {
    int verbose_flags = 0;
    bool stream_block = false;
    bool thinking = true;
    bool keep_special_tokens = false;
};

struct TaskInput {
    ObjectHeader header;
    InputVariant input;
    SamplingConfig sampling;
    OutputOptions output;
    int lora_id = 0;
    std::vector<std::string> guided_choices;
    std::vector<JsonValue> tools;
    std::string visual_encoder_id;
};

struct TaskRequest {
    InputVariant input;
    SamplingConfig sampling;
    OutputOptions output;
    int lora_id;
    std::vector<std::string> guided_choices;
    std::vector<JsonValue> tools;
    std::string visual_encoder_id;
};
```

`TaskRequest` 构造完成后不可修改。限制检查（input bytes、max tokens、verbose 预估内存）在入队前完成，避免明知不可执行的任务占用队列。

### 3.4 Model

```cpp
enum class LoadState { Loading, Ready, Failed, Closing };

struct ModelKey {
    std::string canonical_path;
};

struct SharedModel {
    ModelKey key;
    std::string config_fingerprint;
    std::mutex mutex;
    std::condition_variable load_cv;
    LoadState load_state = LoadState::Loading;
    Status load_status;

    LlamaModelPtr model;
    const llama_vocab *vocab = nullptr; // 生命周期从属于 model
    ModelMetadata metadata;
    std::map<int, LlamaLoraPtr> loras;
    std::unique_ptr<MultimodalBackend> multimodal;
    CountingSemaphore execution_slots;
    std::atomic<int> queued{0};
    std::atomic<int> running{0};
};

struct ModelHandle {
    ObjectHeader header;
    std::shared_ptr<SharedModel> shared_model;
    std::thread::id owner_thread;
    std::mutex task_mutex;
    std::unordered_set<std::shared_ptr<TaskCore>> tasks;
    bool closing = false;
};
```

Model registry 只以 canonical path 作为共享键，以符合“同模型第一次参数有效”的现有注释。第一次加载保存 config fingerprint；后续 fingerprint 不同直接 Load 失败，比静默警告更可控。

### 3.5 Task、Output 与 StreamChunk

```cpp
enum class TaskState {
    Queued, WaitingForModel, Running, Cancelling,
    Completed, Failed, Cancelled, Consumed
};

struct MatrixF32 {
    int rows = 0;
    int cols = 0;
    std::vector<float> values;
    std::vector<const float *> row_ptrs;
    void finalize_row_ptrs();
};

struct TokenResult {
    std::string text;
    std::vector<int> tokens;
    std::vector<int> byte_offsets;
    std::vector<float> probabilities;
    MatrixF32 logits;
    MatrixF32 embeddings;
};

struct InputTokenInfo {
    std::vector<int> tokens;
    std::vector<std::string> text_storage;
    std::vector<const char *> text_ptrs;
};

struct TaskOutput {
    ObjectHeader header;
    TokenResult result;
    InputTokenInfo input;
};

struct StreamChunk {
    TokenResult result;
    std::optional<InputTokenInfo> input;
};

struct TaskCore {
    mutable std::mutex mutex;
    std::condition_variable cv;
    TaskState state = TaskState::Queued;
    std::atomic<bool> cancel_requested{false};
    Status terminal_status;
    TaskRequest request;
    std::shared_ptr<SharedModel> model;
    std::unique_ptr<TaskOutput> output;

    bool streaming = false;
    std::deque<StreamChunk> chunks;
    std::optional<StreamChunk> current_chunk;
    size_t max_queued_chunks = 16;
    std::condition_variable queue_not_full;
};

struct TaskTicket   { ObjectHeader header; std::shared_ptr<TaskCore> core; ModelHandle *owner; };
struct StreamTicket { ObjectHeader header; std::shared_ptr<TaskCore> core; ModelHandle *owner; };
```

worker 只持有 `shared_ptr<TaskCore>` 和 `shared_ptr<SharedModel>`，绝不持有可能被 Unload 释放的裸 `ModelHandle *`。owner 只用于入口校验和从 handle task set 中摘除，摘除动作必须在 handle 销毁前完成。

### 3.6 Runtime 与 Scheduler

```cpp
class Runtime {
public:
    static Runtime &instance();
    Status init();
    void deinit();
    std::shared_ptr<SharedModel> load_or_get(const ModelConfig &, const std::string &);
    Status submit(std::shared_ptr<TaskCore>);

private:
    std::mutex mutex_;
    uint32_t init_ref_count_ = 0;
    uint32_t live_handle_count_ = 0;
    bool backend_initialized_ = false;
    bool shutting_down_ = false;
    ModelRegistry models_;
    Scheduler scheduler_;
};
```

销毁 backend 的条件是 `init_ref_count == 0 && live_handle_count == 0 && scheduler idle`。如果调用方错误地先 inferlet_deinit 再 Unload，Runtime 不再接受新 Load，但延迟真正 backend free，直到最后一个 handle 释放，避免 use-after-free。

---

## 4. 模块边界

### 4.1 `src/api/inferlet_c_api.cpp`

只允许做：

- 公开指针和范围校验；
- 从 opaque 对象转到内部类型；
- 调用 service；
- 设置 out 参数；
- 捕获异常并翻译错误。

禁止在此文件出现生成循环、prompt 拼接、sampler 算法或多模态 shape 逻辑。

统一入口模板：

```cpp
extern "C" int inferlet_some_api(...) noexcept {
    clear_last_error();
    initialize_out_params(...);
    try {
        auto checked = validate(...);
        if (!checked) return publish_error(checked.status());
        return publish_status(service_call(...));
    } catch (const std::bad_alloc &e) {
        return publish_error(INFERLET_E_OUT_OF_MEMORY, e.what());
    } catch (const std::exception &e) {
        return publish_error(INFERLET_E_INTERNAL, e.what());
    } catch (...) {
        return publish_error(INFERLET_E_INTERNAL, "unknown exception");
    }
}
```

### 4.2 `src/engine/llama_engine.*`

这是唯一直接包含 llama.cpp public headers 的业务模块。建议接口：

```cpp
class LlamaEngine {
public:
    Result<LlamaModelPtr> load_model(const ModelConfig &, const std::string &path);
    Result<LlamaContextPtr> create_context(const SharedModel &, const TaskRequest &);
    Result<std::vector<int>> tokenize(const SharedModel &, std::string_view, bool add_special, bool parse_special);
    Result<std::string> detokenize(const SharedModel &, Span<const int>, bool skip_special);
    Result<std::string> token_to_piece(const SharedModel &, int token, bool render_special);
    Status decode_prompt(llama_context *, Span<const int>, int batch_size, const CancelFlag &);
    Result<StepOutput> decode_step(...);
};
```

llama.cpp 的 RAII deleter 也定义在本层。其余源文件不能调用 `llama_model_free`、`llama_free` 等函数。

适配层必须在编译期记录：

```cpp
constexpr const char *INFERLET_LLAMA_COMMIT = "<fixed commit>";
static_assert(sizeof(llama_token) == sizeof(int));
```

### 4.3 `PromptBuilder`

输入是 `TaskRequest + SharedModel`，输出：

```cpp
struct PreparedPrompt {
    std::vector<int> tokens;
    InputTokenInfo verbose_input;
    std::vector<EmbeddedSegment> embeddings; // 纯文本为空
};
```

它负责 chat template、tools、thinking 和多模态占位，但不创建 context、不 decode。

### 4.4 `Sampler`

为确保 probability type=2 可定义且跨 llama.cpp 升级稳定，建议由 Inferlet 维护 sampler pipeline，底层使用 llama.cpp 的 sampler primitive，而不是直接复用 `common_sampler` 的所有默认项。

固定顺序：

```text
原始 logits
-> guided grammar/choice 屏蔽非法 token
-> repetition penalty
-> temperature
-> top-k
-> top-p
-> softmax/抽样（temperature=0 时 argmax）
```

grammar 必须早于 top-k/top-p，否则合法 token 可能先被截断。每选中一个 token 后调用 accept，更新 repetition 与 grammar 状态。

### 4.5 `ResultSink`

```cpp
class ResultSink {
public:
    virtual Status on_input(const InputTokenInfo &) = 0;
    virtual Status on_token(TokenStep &&) = 0;
    virtual void on_finish(Status) = 0;
};

class CollectingSink : public ResultSink { ... };
class StreamingSink  : public ResultSink { ... };
```

同步、异步和流式调用同一个 `Generator::run(request, sink)`。不得维护三套生成循环。

---

## 5. llama.cpp 具体映射

固定 commit 后，在 `doc/llama_compatibility.md` 填写该 commit 的准确签名。映射目标如下：

| Inferlet 能力 | llama.cpp 概念/API |
| --- | --- |
| 全局初始化 | `llama_backend_init` / `llama_backend_free`，需要时加载动态 ggml backend |
| model defaults/load/free | model default params、`llama_model_load_from_file`、`llama_model_free` |
| context | context default params、`llama_init_from_model`、`llama_free` |
| vocab/tokenizer | model vocab、tokenize、token-to-piece；某些版本位于独立 vocab header |
| prompt/generation | batch、`llama_decode`、按需要设置 logits 输出位 |
| logits | 对应输出位置的 logits getter；读取后立即复制 |
| embedding | context 开启 embeddings，对应输出位置 getter；读取后立即复制 |
| sampler | sampler chain/primitives，Inferlet 固定组合与默认值 |
| EOS/special | vocab 的 EOG 判断和 special token API |
| chat | 模型默认 chat template + pinned commit 提供的 template renderer；缺失则失败 |
| LoRA | adapter init/free + per-context adapter set |
| KV/beam | memory/sequence copy、remove/keep；仅 beam executor 使用 |
| 多模态 | pinned commit 的 mtmd 或模型专用 public adapter，隔离在 multimodal 模块 |

不要使用已标记 deprecated 的旧名称。官方头文件已同时保留新旧模型加载名称的时期尤其要避免“能编译但下次升级全部警告/失效”。

### 5.1 模型参数映射

`ModelConfig` 中只影响权重加载的字段形成 `config_fingerprint`：device 配置、GPU layers/tensor split、mmap/mlock、vocab-only、vision backend 和 LoRA 文件集合。`backend_count`、队列上限等调度字段不进入模型权重 fingerprint，但同一 shared model 的第一次调度配置生效。

### 5.2 Context 参数映射

每任务创建 context：

- `n_ctx >= prompt_tokens + max_output_tokens`；实际请求值可由 JSON 上限约束；
- `n_batch/n_ubatch = min(decoder_max_token_count, n_ctx)`；
- `n_seq_max=1`，除非 beam executor；
- 只有 verbose embedding 开启时才启用 embeddings；
- 线程数来自 JSON 或硬件默认；
- abort callback 若 backend 支持则绑定 cancel flag，同时仍在 batch/token 边界主动检查。

创建后查询实际 context 参数并记录 debug 日志，因为 llama.cpp 可能调整请求值。

---

## 6. 关键算法

### 6.1 ChatHistory 展开

每个 `inferlet_chat_history` 元素按以下规则展开：

1. 非空 `system`：只允许整个数组第一次出现；添加 `{role:"system", content:...}`。后续相同 system 可忽略，不同 system 返回 invalid argument。
2. 非空 `query`：添加 user message。
3. 非空 `response`：必须已经有对应 query；添加 assistant message。
4. 最后一个元素允许 response 为 NULL/空，表示等待模型回答。
5. 最终必须以 user/tool 等需要 assistant 继续生成的角色结尾；若以完整 assistant response 结尾，除非配置允许 continuation，否则返回错误。

template renderer 使用 add-generation-prompt 模式。先查询所需 buffer 长度，再分配字符串，禁止猜固定 buffer。

### 6.2 Tokenizer Encode

伪代码：

```text
validate handle/text/text_length/real_token_length
validate text_length bytes are available and UTF-8 valid
required = backend.tokenize(text[0:text_length], add_special, parse_special, null output)
*real_token_length = required
if max_token_length < 0 or (max > 0 and tokens == NULL): invalid argument
encode into temporary vector of required length
copy min(required, max_token_length) tokens to caller
return required > max_token_length ? BUFFER_TOO_SMALL : OK
```

不要将负返回长度直接转成 `size_t`。先检查符号，再分配。

### 6.3 Tokenizer Decode

逐 token 调用 token-to-piece，拼成完整 byte string。`skip_special_tokens=1` 时跳过 special/control token。计算完整 `real_text_length` 后：

- `max_text_length <= 0` 时不写 text；
- 容量大于 0 时最多复制 `max_text_length-1` 字节；
- 向前回退到 UTF-8 合法边界，再补 `\0`；
- 容量不足返回 BUFFER_TOO_SMALL。

### 6.4 Prompt decode

```text
for each batch of prompt tokens:
    if cancelled: return CANCELLED
    construct batch positions [processed, ...]
    only request logits for final prompt token
    ret = llama_decode(context, batch)
    if ret != 0: return INFERENCE error with ret
```

若 prompt 为空：优先使用模型 BOS；模型没有 BOS 且不能空 decode 时返回 invalid argument。不要凭空使用 token 0。

### 6.5 单 token 生成

```text
for step in [0, max_output_tokens):
    check cancellation
    raw_logits = copy logits corresponding to previous decode output
    if verbose logits: reserve/check budget, append raw_logits

    if probability_type == 1:
        p_before = stable_softmax(raw_logits)[eventual_selected_token]

    candidates = raw_logits
    apply grammar mask, repetition, temperature, top-k, top-p
    normalize candidates
    selected = greedy_or_sample(candidates, rng)

    compute p_before for selected if requested
    p_after = normalized_candidates[selected] if type == 2
    sampler.accept(selected)

    if selected is EOG:
        finish without adding selected to public result

    decode selected token with logits requested for next step
    if verbose embedding: copy selected token embedding now
    convert selected to piece according to special-token mode
    append token/piece/offset/probability/verbose data to sink

finish with OK when max tokens reached
```

`stable_softmax` 使用减最大值法并以 double 累加，处理 NaN/Inf；若候选集为空或总概率无效，返回 inference error，不回退到随机 token。

注意：p_before 需要先知道 selected，因此可以先保存 raw logits，在 selected 后只计算 selected 的 softmax 分子/分母，不必额外保存一份概率向量。

### 6.6 文本与 offset

- `text` 是所有可见 token piece 的字节拼接。
- 在追加 piece 前记录当前 `text.size()` 为该 token offset。
- token 不可见或 piece 为空时仍保留 token，offset 等于当前末尾。
- offset 使用 `int` 返回，若累计字节数超过 `INT_MAX`，任务失败为资源/范围错误。
- 流式 sink 累积 token pieces，直到字节序列是合法 UTF-8 或任务结束，再形成一个 chunk。
- 若结束时仍有非法 UTF-8 尾部，按严格模式返回推理错误；不自动插入替换字符，以免 tokens 与 text 不一致。

### 6.7 Verbose 数据

提交前估算最坏内存：

```text
logits_bytes    = max_output_tokens * n_vocab * 4
embedding_bytes = max_output_tokens * n_embd  * 4
```

任一乘法先做 checked multiplication。超过 JSON 的 `limits.max_verbose_bytes` 直接拒绝任务。实际运行中每次 append 仍检查累计预算。

最后一个 token 的 embedding 只有在该 token decode 后才存在；即使达到 max tokens，也必须完成这次 decode/copy 后才能结束。EOG 被排除于结果，因此不需要保存其 embedding。

### 6.8 GuidedChoices

将 choices 转为模型 tokenizer/grammar 可表达的有限语言：

- JSON/GBNF 字符串做完整转义；
- exact choice 后只允许 EOG；
- common prefix 通过 trie 合并；
- 空 choice 是否允许必须明确，建议拒绝；
- 与 tools 同时配置第一版返回 unsupported，避免两个约束器冲突。

grammar 状态每 task 独立，禁止共享。

### 6.9 Beam search

beam size=1 使用普通生成器。大于 1 使用独立 executor：

1. prompt 只 decode 一次；
2. 扩展首轮 top `beam_size` 候选；
3. 为每个 beam 分配 sequence ID，并复制 prompt KV sequence；
4. 每轮对各 beam decode，累计 `sum(log(prob))`；
5. 每轮从所有扩展中保留全局 top beams；
6. finished beam 不再扩展，直到全部结束或 max tokens；
7. 使用 run config 的 length penalty（默认 1.0）选择最终 beam；
8. 清理所有非最终 sequence。

beam 模式与随机 temperature/top-p 的组合容易语义不清。v1 规定 beam>1 时只使用 repetition + grammar + log-softmax，忽略随机 sampler 并记录该契约；更推荐发现非默认随机参数时返回 invalid argument。

### 6.10 MultiModal

内部先规范化成：

```cpp
struct MediaTensor {
    MediaType type;
    DataType dtype;
    std::vector<std::byte> owned_data;
    int grid_t, grid_h, grid_w;
    float seconds_per_grid_t;
    float fps;
    FrameInfo frames;
    CachePolicy cache;
};

struct MessagePart {
    std::variant<std::string, MediaTensor> value;
};
```

所有 size 公式使用 checked multiplication。raw uint8 的通道数、patch size 和 temporal patch size 从模型 backend metadata 获取，不能写死。

规范化前执行以下字段规则：

- `inferlet_multimodal` 中 feature 非空时 size 必须大于 0；prompt 可为空，数组顺序固定为 text 后跟 vision feature。
- ChatContent type=`text` 时将 data 当作 NUL 结尾 UTF-8 字符串深拷贝，size/grid/dtype 必须为 0。
- type=`image`/`video` 时 data 非空、size 大于 0，grid 三维均为正；dtype 只接受 1～4，同一 task 必须一致。
- feature dtype 1/2/3 的 size 必须等于 `grid_t*grid_h*grid_w*n_embd*element_size`；所有因子先做 checked multiplication。
- uint8 的 shape 由视觉模型 metadata 校验；不得只根据传入 size 猜通道数。
- `frame_num < 0` 非法；大于 0 时 frame_index 可为空（生成从 0 开始的连续索引）或指向至少 frame_num 个 int。
- packing_num=0 规范成 1，非零时只接受 1～6；fps=0 规范成 24，负值非法。
- cache_key 为空时不缓存；非空时要求 cache_times>=2。
- 当前版本所有 `reserved_p` 必须为 NULL、`reserved_i` 必须为 0。不要保存或解引用未知 reserved 指针；将来启用某字段时通过 ABI version 明确解释。

视觉缓存键实际为：

```text
model_fingerprint + visual_encoder_id + user_cache_key + dtype + shape
```

第一次插入保存 metadata/hash；后续同 key 不一致返回 invalid argument。`cache_times` 是成功取用次数预算，不是存活秒数。达到次数后从 map 移除；仍被在途任务引用的数据由 shared_ptr 延迟释放。

---

## 7. 并发和销毁设计

### 7.1 锁顺序

固定全局锁顺序，任何代码不得反向获取：

```text
Runtime mutex
-> ModelRegistry mutex
-> ModelHandle task mutex
-> SharedModel mutex
-> TaskCore mutex
```

执行 llama decode 时不持有以上管理锁。stream sink 等待 queue_not_full 时只持有 TaskCore mutex，并且 CV wait 会释放它。

### 7.2 Scheduler

- 固定数量 worker；数量来自全局 JSON 或硬件默认。
- 每个 SharedModel 有 execution semaphore，容量为 backend_count。
- worker 取 task 后先等待 model load 终态，再取得 model execution slot。
- 为避免一个模型占满全局 worker，等待 semaphore 不应长期占用 worker。可使用 ready queue：model 有 slot 时才将 task 转入 runnable queue。
- 最小正确实现允许 worker 等 slot，但要将全局 worker 数设为所有 backend count 之和，并在后续优化 ready queue。

### 7.3 Async load

同一 canonical path 并发 Load 时：

1. registry 第一个请求创建 Loading `SharedModel` 并启动 load job；
2. 后续相同 fingerprint 共享该对象；不同 fingerprint 立刻失败；
3. 同步调用等待 `load_cv`；异步调用立即返回 handle；
4. load 成功设置 Ready，失败保存完整 Status；两者都 notify_all；
5. Run 在 WaitingForModel 状态等待；失败时以相同 load status 终止；
6. `inferlet_check_backend_active` 在 Loading 时返回 active=1，在 Failed 时返回 MODEL_LOAD。

### 7.4 Cancel

- Queued：scheduler 从队列跳过并转 Cancelled。
- WaitingForModel：不取消共享的模型加载，只取消该 task。
- Running：设置 atomic flag；CPU abort callback 和 token/batch 边界检查。
- GPU backend 可能无法立即中断已提交 kernel，因此 Cancel 是协作式，不保证瞬时返回。
- Cancel 与 Completed 竞争时，持 TaskCore mutex 的第一个终态转换获胜；终态不可覆盖。

### 7.5 Unload

```text
validate owner thread
mark handle.closing
snapshot all handle tasks
request cancel for each
wait every worker release TaskCore/SharedModel usage
invalidate and remove all outstanding tickets
clear handle task set
remove handle from live registry
release SharedModel reference
delete ModelHandle
try Runtime deferred shutdown
```

因为 Unload 返回 void，它必须完成清理后再返回。不要 detached thread。

### 7.6 Stream 背压

默认最多 16 个 chunks。worker 生产第 17 个时等待 `queue_not_full`；消费者每 pop 一个 notify。取消和 Unload 必须同时 notify `cv` 与 `queue_not_full`，否则 worker 可能永久阻塞。

---

## 8. Run Config 设计

建议第一版 schema：

```json
{
  "schema_version": 1,
  "model": {
    "context_size": 4096,
    "n_batch": 512,
    "n_ubatch": 512,
    "threads": 0,
    "threads_batch": 0,
    "mmap": true,
    "mlock": false,
    "gpu_layers": 0
  },
  "scheduler": {
    "worker_count": 0,
    "queue_capacity": 256,
    "stream_queue_chunks": 16
  },
  "sampling": {
    "seed": 0,
    "repeat_last_n": 64,
    "beam_length_penalty": 1.0
  },
  "limits": {
    "max_input_bytes": 67108864,
    "max_multimodal_bytes": 536870912,
    "max_verbose_bytes": 536870912,
    "max_visual_cache_bytes": 1073741824
  },
  "logging": {
    "level": "info"
  }
}
```

规则：

- 必须有 `schema_version`；不支持的版本加载失败。
- 类型错误、范围错误、未知顶层字段默认失败，避免拼写错误被忽略。
- 相对路径以 run config 文件所在目录解析。
- setter 可表达的字段由 setter 覆盖 JSON；其余字段只来自 JSON。
- 解析后生成不可变 EffectiveConfig，并打印脱敏后的摘要。

---

## 9. 构建和发布设计

### 9.1 普通开发构建

JSON 统一使用一份固定版本的 `nlohmann/json`（MIT、header-only）或团队指定的等价解析器，并放在 `third_party`/包管理锁文件中。不要一部分使用 llama.cpp common 的 JSON 类型、另一部分自行解析；`SetTools` 和 run config 必须共用同一个严格 parser。

顶层 CMake 结构：

```cmake
cmake_minimum_required(VERSION 3.20)
project(Inferlet VERSION 0.1.1 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

add_subdirectory(third_party/llama.cpp EXCLUDE_FROM_ALL)
find_package(Threads REQUIRED)

add_library(Inferlet STATIC ${INFERLET_SOURCES})
add_library(Inferlet::Inferlet ALIAS Inferlet)
target_include_directories(Inferlet
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:include>
  PRIVATE src)
target_link_libraries(Inferlet PRIVATE llama Threads::Threads)
```

实际 llama.cpp CMake 选项必须在固定 commit 后核对并显式设置，关闭 examples/tests/server，避免上游默认变化污染产物。

### 9.2 Fat archive

普通 STATIC target 不会吸收它依赖的静态 archive。发布任务应：

1. 构建 `libinferlet_wrapper.a`、llama 和 ggml 相关 archives；
2. 列出最终需要合并的 archive，不能用不确定 glob；
3. Linux 用 MRI 脚本 `create/addlib/save/end` 合并；
4. `ranlib libinferlet.a`；
5. 用 `nm -u` 检查剩余未定义符号只属于允许的系统/GPU runtime；
6. clean consumer 链接真实可执行文件。

不要将 `/usr/lib` 中的 libc/libstdc++ 等系统 archive 合入项目产物。

### 9.3 安装

- 安装 public header 和最终 fat archive。
- CMake imported target 传递 Threads、`dl`、`m` 及所选 GPU runtime。
- `.pc` 的 `Libs.private` 记录静态链接依赖。
- 安装 llama.cpp MIT license 和项目 third-party notices。

---

## 10. 逐步实施清单

严格按顺序执行。每一步只在“完成检查”通过后进入下一步。

### Step 1：冻结头文件契约

创建/修改：

- `include/inferlet.h`
- `doc/api_contract.md`
- `tests/abi/header_c.c`
- `tests/abi/header_cpp.cpp`

实施：完成第 2 节修正；为 64 个函数建立符号清单 `tests/abi/expected_symbols.txt`。

完成检查：C/C++ 头文件独立编译；结构体 `sizeof/offsetof` 基线测试通过；review 确认 task/stream 回收语义。

### Step 2：固定 llama.cpp 并搭建 CMake

创建/修改：

- `.gitmodules`、`third_party/llama.cpp`
- `CMakeLists.txt`
- `cmake/InferletOptions.cmake`
- `doc/third_party.md`
- `doc/llama_compatibility.md`

实施：选择经过 CPU smoke test 的 commit，记录完整 40 位 hash；关闭无关上游 targets；建立空的 `Inferlet` 静态目标。

完成检查：从 clean build 目录完成 Debug/Release；打印并保存 llama commit；禁止联网拉取构建依赖。

### Step 3：基础设施和 C API 空壳

创建：

- `src/support/{status,last_error,logging,object_registry,validation}.*`
- `src/api/inferlet_c_api.cpp`
- `tests/unit/test_validation.cpp`
- `tests/abi/test_symbols.cmake`

实施：错误码、TLS last error、异常边界、live object registry。为全部函数先提供实现，未实现能力返回 unsupported。

完成检查：`nm` 与 64 符号基线相同；随机 NULL/错误类型句柄 smoke test 不崩溃；ASan 通过。

### Step 4：Runtime 生命周期

创建：

- `src/core/runtime.*`
- `src/core/scheduler.*`（先只实现 start/stop）
- `tests/unit/test_runtime.cpp`

实施：Init 引用计数、deferred shutdown、llama backend RAII；禁止在持 runtime lock 时调用耗时 llama API。

完成检查：成千上万次嵌套 Init/inferlet_deinit，无负计数、double free；未 Init 的 Load 明确失败。

### Step 5：Parameter 与 JSON

创建：

- `src/core/model_config.*`
- `src/support/json_config.*`
- `tests/unit/test_model_config.cpp`

实施：所有 Parameter API、deep copy、invalid state、JSON schema/优先级/fingerprint。

完成检查：每个 setter 的默认、边界、重复调用有测试；坏 JSON 和不存在文件不泄漏。

### Step 6：LlamaEngine、模型和 tokenizer

创建：

- `src/engine/llama_engine.*`
- `src/engine/token_codec.*`
- `src/core/model_registry.*`
- `src/core/model_handle.*`
- `tests/integration/test_tokenizer.cpp`

实施：同步 load/free、vocab-only、canonical path、共享 registry、Encode/Decode。

完成检查：与 pinned llama.cpp tokenize 工具逐 token 对比；buffer too small、UTF-8 和 special token 用例通过；同模型只加载一次。

### Step 7：TaskInput

创建：

- `src/core/task_input.*`
- `tests/unit/test_task_input.cpp`

实施：所有 setter、input variant、last-wins、recursive deep copy、限制/溢出校验、immutable request snapshot。

完成检查：setter 后立即销毁源内存仍可生成 request；重复替换输入在 ASan 下无泄漏。

### Step 8：PromptBuilder

创建：

- `src/engine/prompt_builder.*`
- `tests/unit/test_prompt_builder.cpp`

实施：text/tokens/chat；input verbose token pieces；暂不接多模态。tools/guided 只做结构校验并标记能力。

完成检查：至少两个模型模板 golden test；无模板模型明确失败；空 prompt/BOS 行为固定。

### Step 9：Sampler

创建：

- `src/engine/sampler.*`
- `tests/unit/test_sampler.cpp`

实施：固定 pipeline、seed、概率 before/after、repetition history、EOG。用人工 logits 向量验证数学结果。

完成检查：greedy、top-k、top-p、temperature、penalty 的边界测试；固定 seed 重现；概率和为 1（容差内）。

### Step 10：同步生成与 TaskOutput

创建：

- `src/engine/generator.*`
- `src/core/task_output.*`
- `src/engine/result_sink.*`
- `tests/integration/test_sync_generation.cpp`

实施：prompt decode、token loop、CollectingSink、text/tokens/offset/probability、context 销毁。

完成检查：真实小 GGUF 生成；同步 tokens decode 与 text 一致；max tokens/EOG/context overflow/推理错误全部覆盖。

### Step 11：Verbose 数据

修改 generator/output；新增 `test_verbose_output.cpp`。

实施：input tokens、raw logits、embedding、MatrixF32 finalize、预算检查。

完成检查：shape、首尾时间步与直接 llama 调用一致；未开启 getter 返回 NULL；超过预算在分配巨量内存前失败。

### Step 12：Scheduler 和普通异步任务

创建：

- `src/core/inference_task.*`
- `tests/unit/test_task_state.cpp`
- `tests/integration/test_async.cpp`

实施：worker queue、model semaphore、inferlet_run_async/Query/Wait/Cancel、handle task set、inferlet_check_backend_active。

完成检查：queued/running cancel、Query/Wait 竞争、重复 consume、Unload 竞争；TSan 和 10k task 压力测试通过。

### Step 13：异步模型加载

修改 registry/runtime/model handle；新增 `test_async_load.cpp`。

实施：第 7.3 节状态机、共享 load job、失败传播、同步/异步参数。

完成检查：不存在/损坏模型的异步错误可由 Check 和 Run 观察；Unload loading handle 不死锁。

### Step 14：流式推理

创建：

- `src/core/stream_sink.*`
- `tests/integration/test_stream.cpp`

实施：StreamTicket、有界 chunk queue、block mode、current chunk getters、UTF-8 聚合、EOF 自动回收。

完成检查：所有 chunk 拼接与同步输出一致；慢消费者背压、取消、空 piece、EOF 后禁止使用通过 ASan/TSan。

### Step 15：Guided、Tools、Thinking 和 LoRA

创建：

- `src/engine/guided_choice.*`
- `src/engine/model_prompt_adapter.*`
- `src/engine/lora_registry.*`
- 对应 integration tests

实施顺序：guided choices -> thinking -> tools -> LoRA。每一项建立模型能力表。

完成检查：支持模型成功；不支持模型返回 unsupported；非法 tool JSON 在 setter 阶段失败；LoRA ID 并发隔离。

### Step 16：Beam search

创建 `src/engine/beam_search.*` 和 deterministic tests。按第 6.9 节实现，不修改普通 sampler generator。

完成检查：人工小词表可手算最终 beam；KV sequence 无残留；beam=1 与普通 greedy 完全相同。

### Step 17：多模态

创建：

- `src/multimodal/multimodal_backend.*`
- `src/multimodal/mtmd_backend.*`
- `src/multimodal/visual_cache.*`
- `tests/integration/test_multimodal.cpp`

实施顺序：单图 raw uint8 -> legacy feature -> multimodal chat -> 多图 -> 视频 -> 缓存 -> 多视觉编码器。

完成检查：每种输入 shape 都有正反测试；同输入 cache hit/miss 输出一致；cache_times 和内存上限正确。

### Step 18：Fat archive、安装和发布

创建：

- `cmake/InferletFatArchive.cmake`
- `cmake/InferletConfig.cmake.in`
- `inferlet.pc.in`
- `examples/{sync,async,stream,tokenizer}.cpp`
- `tests/consumer/{c,cpp}/`

实施：合并 archive、安装/export、依赖传递、license、符号/ABI baseline。

完成检查：在源码树以外、清空构建环境后安装；C/C++ consumer 使用安装前缀构建并运行；`nm -u` 只剩允许的系统依赖。

---

## 11. 必须具备的测试用例

### API 与所有权

- `API-001`：每个 Deallocate(NULL) 不崩溃。
- `API-002`：把 Parameter 当 TaskInput 传入返回 invalid handle。
- `API-003`：setter 深拷贝字符串、tokens、chat、feature。
- `API-004`：Run 后立即释放 TaskInput，任务仍正常完成。
- `API-005`：Output getter 指针在 Deallocate 前地址和内容稳定。
- `API-006`：已消费 ticket 再使用不会读已释放内存；Debug 至少报告 invalid handle。

### 生命周期和并发

- `LIFE-001`：Init 两次、inferlet_deinit 两次，只初始化/释放 backend 一次。
- `LIFE-002`：先 inferlet_deinit 后 Unload，backend 延迟释放。
- `LIFE-003`：Unload 自动取消 queued/running/stream tasks。
- `ASYNC-001`：Query 未完成返回 0 + NULL。
- `ASYNC-002`：Wait 与完成无遗漏唤醒。
- `ASYNC-003`：取消/完成竞争只产生一个终态。
- `ASYNC-004`：两个 handle 共享 model 但输出状态不互相污染。
- `STREAM-001`：非阻塞空串、有效 chunk、最终 NULL 顺序正确。
- `STREAM-002`：阻塞模式不无限返回空串。
- `STREAM-003`：背压下取消能唤醒 producer。

### 推理正确性

- `GEN-001`：greedy 与 pinned llama.cpp reference 相同。
- `GEN-002`：固定 seed sampling 可重现。
- `GEN-003`：offset 对 ASCII、多字节 UTF-8、byte fallback、空 special piece 正确。
- `GEN-004`：probability type 1/2 与手算 logits 相同。
- `GEN-005`：同步与流式拼接 text/tokens/probability 相同。
- `GEN-006`：max output=1 时最后 token embedding 仍存在。
- `GEN-007`：EOG 不进入结果数组。
- `GEN-008`：context 不足在 decode 前失败。

### 边界与故障

- 所有 length 的 -1、0、`INT_MAX`；所有 size 乘法溢出。
- 空 prompt、空 tokens、空 history、NULL 数组与非零 length。
- 损坏 GGUF、没有 chat template、非法 LoRA、非法 device。
- verbose 最坏内存超过限制。
- worker 内存分配失败、decode 返回错误、stream consumer 永不读取。

---

## 12. 实施时的每日检查规则

每完成一个 API 或模块：

1. 先更新/确认 `api_contract.md`，不要从当前实现反推契约。
2. 加入正常、非法参数、生命周期三个方向的测试。
3. 运行 Debug 单测和相关真实模型集成测试。
4. 涉及内存运行 ASan/UBSan；涉及线程运行 TSan。
5. 检查 C API 是否有异常越界、out 参数是否在失败时初始化。
6. 检查是否保存了调用方裸指针或 llama.cpp 临时 buffer。
7. 检查新功能在不支持模型上是否明确失败。
8. 更新 `llama_compatibility.md` 和支持矩阵。

建议常用命令最终固化为 CMake presets：

```sh
cmake --preset debug
cmake --build --preset debug -j
ctest --preset debug --output-on-failure

cmake --preset asan
cmake --build --preset asan -j
ctest --preset asan --output-on-failure

cmake --preset tsan
cmake --build --preset tsan -j
ctest --preset tsan --output-on-failure
```

---

## 13. 开始实施前的最终决策清单

下面各项必须在 Step 1 review 中打勾：

- [ ] 调用方是否需要纯 C；若需要，已完成 typedef 修正。
- [ ] 是否允许新增 `GetLastError` 和 stream 显式释放 API。
- [ ] EOS、special token、offset、probability、logits、embedding 语义已确认。
- [ ] 普通 task 和 stream cancel 后的回收语义已确认。
- [ ] 默认采样参数已确认并写入测试。
- [ ] async load 的失败观察方式已确认。
- [ ] 同路径不同参数选择“失败”而非“首个参数静默生效”已确认。
- [ ] llama.cpp commit 已固定并记录。
- [ ] v1 首发平台和至少一个文本测试模型已确定。
- [ ] 多模态目标模型族及 mmproj/feature 格式已确定。
- [ ] fat archive 对“只链接一个项目库”的边界已确认：系统和 GPU runtime 仍由 CMake/pkg-config 传递。

这些决策完成后，从 Step 2 开始可以按照本文逐项实施。若某项需求变化，先修改契约和测试，再修改实现。

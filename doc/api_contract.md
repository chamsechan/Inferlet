# Inferlet 公共 API 契约规范

## 1. 概述与设计原则

Inferlet 对外提供纯 C ABI 头文件 [`include/inferlet.h`](../include/inferlet.h)。
本规范定义公开函数、数据结构、生命周期、数据所有权及错误处理的确定契约。

项目统一使用以下名称：

| 用途 | 名称 |
|---|---|
| 公共头文件 | `inferlet.h` |
| C 函数与类型 | `inferlet_*`，使用小写下划线命名 |
| 错误码、标志与构建配置 | `INFERLET_*` |
| 静态库 | `libinferlet.a` |
| CMake 包与目标 | `find_package(Inferlet CONFIG REQUIRED)`、`Inferlet::Inferlet` |
| pkg-config 包 | `inferlet` |
| 文本测试模型 | `INFERLET_TEST_MODEL`、`INFERLET_TEST_MODEL_DIR` |
| 图像测试模型和配置 | `INFERLET_TEST_VISION_MODEL`、`INFERLET_TEST_VISION_CONFIG` |

本次统一命名更改了公开符号，调用方需要更新头文件、标识符及链接配置并重新编译；不提供旧名称别名。函数参数、结构体布局和错误码数值保持不变。

```cmake
find_package(Inferlet CONFIG REQUIRED)
target_link_libraries(app PRIVATE Inferlet::Inferlet)
```

```sh
cc app.c $(pkg-config --cflags --libs --static inferlet) -o app
```

- **C/C++ 双兼容**：头文件符合 C11 与 C++17 标准，结构体使用 `typedef struct Name { ... } Name;` 形式。
- **纯 C 边界**：严禁在公共头文件中出现 C++ STL、`std::string`、`std::vector`、llama.cpp 或 ggml 内部类型。
- **异常完全屏蔽**：所有导出 C API 函数使用 `noexcept` / `try-catch (...)` 保护，绝不允许 C++ 异常跨越 ABI。
- **内存深拷贝**：所有 setter 对传入的字符串、数组及多模态数据执行深拷贝。调用方在 setter 返回后即可安全释放或修改原内存。
- **提交快照机制**：`inferlet_run`、`inferlet_run_async`、`inferlet_run_stream` 接收到 `task_in` 时制作不可变的内部 `TaskRequest`，提交后调用方可复用、修改或释放 `task_in`。

---

## 2. 导出宏与错误码

### 2.1 导出宏

```c
#ifndef INFERLET_H_INCLUDED
#define INFERLET_H_INCLUDED

#if defined(_WIN32) || defined(__CYGWIN__)
#  if defined(INFERLET_BUILDING_LIBRARY) && defined(INFERLET_SHARED)
#    define INFERLET_API __declspec(dllexport)
#  elif defined(INFERLET_SHARED)
#    define INFERLET_API __declspec(dllimport)
#  else
#    define INFERLET_API
#  endif
#else
#  if defined(INFERLET_BUILDING_LIBRARY) && defined(INFERLET_SHARED)
#    define INFERLET_API __attribute__((visibility("default")))
#  else
#    define INFERLET_API
#  endif
#endif
```

### 2.2 错误码枚举

所有返回 `int` 的 API 均遵循：`0` 表示成功，负值表示错误。

```c
typedef enum inferlet_error_code
{
    INFERLET_OK                  =  0,  /* 成功 */
    INFERLET_E_INVALID_ARGUMENT  = -1,  /* 无效参数（指针为空、越界、非法 UTF-8 等） */
    INFERLET_E_INVALID_HANDLE    = -2,  /* 无效句柄（非对应类型对象、未初始化对象） */
    INFERLET_E_INVALID_STATE     = -3,  /* 状态错误（未 Init、重复关闭、已消费等） */
    INFERLET_E_IO                = -4,  /* 文件/IO 错误（配置文件读取失败、文件不存在等） */
    INFERLET_E_MODEL_LOAD        = -5,  /* 模型加载失败（GGUF 损坏、参数冲突等） */
    INFERLET_E_INFERENCE         = -6,  /* 推理执行错误（上下文溢出、decode 失败等） */
    INFERLET_E_UNSUPPORTED       = -7,  /* 不支持的操作或模型能力 */
    INFERLET_E_CANCELLED         = -8,  /* 任务已取消 */
    INFERLET_E_BUFFER_TOO_SMALL  = -9,  /* 输出缓冲区容量不足 */
    INFERLET_E_OUT_OF_MEMORY     = -10, /* 内存分配失败（std::bad_alloc） */
    INFERLET_E_INTERNAL          = -11  /* 内部未知异常或未捕获错误 */
} inferlet_error_code;
```

---

## 3. 对象生命周期与句柄矩阵

| 公开句柄 | 内部对应类型 | 创建 API | 销毁 API / 规则 |
|---|---|---|---|
| `param` (`void *`) | `Parameter` | `inferlet_parameter_allocate` | `inferlet_parameter_deallocate` |
| `handle` (`void *`) | `ModelHandle` | `inferlet_load_from_file` | `inferlet_unload` |
| `task_in` (`void *`) | `TaskInput` | `inferlet_task_input_allocate` | `inferlet_task_input_deallocate` |
| `task` (`void *`) | `TaskTicket` | `inferlet_run_async` | 首次交付结果的 `inferlet_query_task` / `inferlet_wait_task`；或取消后由二者回收 |
| `task_stream` (`void *`) | `StreamTicket` | `inferlet_run_stream` | `inferlet_get_stream_text` 返回 NULL 时自动销毁；或 `inferlet_cancel_task` 显式销毁 |
| `task_out` (`void *`) | `TaskOutput` | 由 `inferlet_run` / `inferlet_query_task` / `inferlet_wait_task` 输出 | `inferlet_task_output_deallocate` |

- **Deallocate(NULL) 规则**：所有 `Deallocate(NULL)` 和 `Unload(NULL)` 均为安全无操作 (no-op)。
- **单一归属**：同一 `task` ticket 只能被一个消费线程调用 `inferlet_query_task` 或 `inferlet_wait_task`，一旦交付结果或确认取消/失败，ticket 即被销毁，调用方不可再次访问。
- **流式终态回收**：`inferlet_get_stream_text` 一旦返回 `NULL`，表明流式任务已终结，内部已自动回收 `task_stream`，调用方不可再次传入。
- **生成文本完整性**：同步和异步结果只在文本为完整有效 UTF-8 时交付；输出 token 上限截断了多字节字符时，任务以 `INFERLET_E_INFERENCE` 失败。流式任务保留此前完整的 chunk，收尾若存在不完整字符则记录失败。
- **流式错误可观察性**：正常结束、失败与取消均通过 `inferlet_get_stream_text == NULL` 表示。现有公共 ABI 没有错误查询函数；需要获取明确的完成错误码时，应使用 `inferlet_run` 或 `inferlet_run_async` 配合 `inferlet_wait_task` / `inferlet_query_task`。
- **句柄安全性**：内部使用 `LiveObjectRegistry` 和带魔数的 `ObjectHeader` 校验句柄合法性。

---

## 4. 数据结构规范

### 4.1 `inferlet_chat_history`
```c
typedef struct inferlet_chat_history
{
    const char *system;   /* 系统提示，仅在首轮出现，后续为 NULL 或空 */
    const char *query;    /* 用户提问 */
    const char *response; /* 助手回复，连续对话当前轮为空/NULL */
} inferlet_chat_history;
```

### 4.2 `inferlet_multimodal`
```c
typedef struct inferlet_multimodal
{
    const char *prompt_text; /* 提示语 */
    const void *feature;     /* 特征数据指针 (FP32) */
    void *reserved_p[2];     /* 保留字段，必须设为 NULL */
    int size;                /* 特征字节数 */
    int reserved_i[7];       /* 保留字段，必须设为 0 */
} inferlet_multimodal;
```

### 4.3 `inferlet_multimodal_chat_content_ext`
```c
typedef struct inferlet_multimodal_chat_content_ext
{
    int *frame_index;       /* 帧索引 */
    const char *language;   /* 语言 */
    void *reserved_p[6];    /* 保留字段，必须设为 NULL */
    int frame_num;          /* 帧数量 */
    int packing_num;        /* 打包数量 1-6 */
    int slice_num;          /* 切分行数 */
    int reserved_i[9];      /* 保留字段，必须设为 0 */
} inferlet_multimodal_chat_content_ext;
```

### 4.4 `inferlet_multimodal_chat_content`
```c
typedef struct inferlet_multimodal_chat_content
{
    const char *type;        /* "text", "image", "video" */
    const void *data;        /* 字符串或多模态数据 */
    const char *cache_key;   /* 缓存 key */
    inferlet_multimodal_chat_content_ext *ext; /* 扩展信息 */
    int size;                /* 字节数 */
    int grid_t, grid_h, grid_w; /* patch 数量 */
    int data_type;           /* 0: undef, 1: float, 2: half, 3: bf16, 4: uint8 */
    float second_per_grid_t; /* 帧间隔 */
    int cache_times;         /* 缓存使用次数 >= 2 */
    float fps;               /* 帧率 (默认 24) */
} inferlet_multimodal_chat_content;
```

### 4.5 `inferlet_multimodal_chat_history`
```c
typedef struct inferlet_multimodal_chat_history
{
    const char *role;                        /* "system", "user", "assistant" */
    inferlet_multimodal_chat_content *content; /* 对话内容 */
    int length;                              /* 数量 */
    int reserved_i;                          /* 保留，必须设为 0 */
} inferlet_multimodal_chat_history;
```

---

## 5. 全部 64 个导出 API 详述

### 5.1 运行环境与模型管理 (5 个函数)
1. `void inferlet_init(void)`: 全局初始化。支持引用计数，线程安全。
2. `void inferlet_deinit(void)`: 全局反初始化。引用计数归零且无存活句柄时释放后台资源。
3. `void *inferlet_load_from_file(const char *file_name, void *param)`: 加载模型，成功返回 handle，失败返回 NULL。
4. `void inferlet_unload(void *handle)`: 卸载模型，取消所有正在进行的相关任务并安全释放。
5. `int inferlet_check_backend_active(void *handle, int *is_active)`: 查询当前 handle 是否有任务在执行或加载中。

### 5.2 推理执行与控制 (6 个函数)
6. `int inferlet_run(void *handle, void *task_in, void **task_out)`: 同步执行推理。
7. `int inferlet_run_async(void *handle, void *task_in, void **task)`: 异步执行推理，成功返回 0 并写 `*task`。
8. `int inferlet_run_stream(void *handle, void *task_in, void **task_stream)`: 流式执行推理，成功返回 0 并写 `*task_stream`。
9. `int inferlet_wait_task(void *handle, void *task, void **task_out)`: 阻塞等待异步任务完成并交付 `*task_out`，自动回收 `task`。
10. `int inferlet_query_task(void *handle, void *task, void **task_out)`: 非阻塞查询异步任务状态。未完成时写 NULL 返回 0；完成时写 `*task_out` 返回 0 并回收 `task`；失败返回负数并回收 `task`。
11. `int inferlet_cancel_task(void *handle, void *task)`: 取消任务。

### 5.3 任务输入构建 `TaskInput` (23 个函数)
12. `void *inferlet_task_input_allocate(void)`: 分配 TaskInput。
13. `void inferlet_task_input_deallocate(void *task_in)`: 释放 TaskInput。
14. `int inferlet_task_input_set_prompt_text(void *task_in, const char *prompt)`: 设置提示文本。
15. `int inferlet_task_input_set_prompt_tokens(void *task_in, int *tokens, int length)`: 设置提示 token 序列。
16. `int inferlet_task_input_set_chat_history(void *task_in, inferlet_chat_history history[], int length)`: 设置历史对话。
17. `int inferlet_task_input_set_multimodal(void *task_in, inferlet_multimodal input[], int length)`: 设置多模态特征。
18. `int inferlet_task_input_set_multimodal_chat_history(void *task_in, inferlet_multimodal_chat_history history[], int length)`: 设置多模态对话历史。
19. `int inferlet_task_input_set_low_rank_adapter_id(void *task_in, int lora_id)`: 设置 LoRA ID。
20. `int inferlet_task_input_set_beam_size(void *task_in, int beam_size)`: 设置 beam 大小。
21. `int inferlet_task_input_set_top_k(void *task_in, int top_k)`: 设置 Top-K。
22. `int inferlet_task_input_set_top_p(void *task_in, float top_p)`: 设置 Top-P。
23. `int inferlet_task_input_set_temperature(void *task_in, float temperature)`: 设置温度。
24. `int inferlet_task_input_set_repetition_penalty(void *task_in, float repetition_penalty)`: 设置重复惩罚。
25. `int inferlet_task_input_set_max_output_tokens(void *task_in, int max_tokens)`: 设置最大输出 token 数。
26. `int inferlet_task_input_set_probability_type(void *task_in, int type)`: 设置输出概率类型 (0: 无, 1: 采样前, 2: 采样后)。
27. `int inferlet_task_input_set_verbose_data_flag(void *task_in, int flag)`: 设置 verbose 掩码。
28. `int inferlet_task_input_set_stream_block_mode(void *task_in, int mode)`: 设置流式阻塞模式 (0: 非阻塞, 1: 阻塞)。
29. `int inferlet_task_input_set_thinking_mode(void *task_in, int mode)`: 设置思考模式 (Qwen3 支持)。
30. `int inferlet_task_input_set_special_token_mode(void *task_in, int keep)`: 设置文本是否保留特殊 token。
31. `int inferlet_task_input_set_guided_choices(void *task_in, const char **choices, int num)`: 设置 guided choices 限制生成。
32. `int inferlet_task_input_set_tools(void *task_in, const char **tools, int num)`: 设置 tools 函数列表。
33. `int inferlet_task_input_set_visual_encoder(void *task_in, const char *id)`: 设置视觉编码器 ID。

### 5.4 任务输出读取 `TaskOutput` (8 个函数)
34. `void inferlet_task_output_deallocate(void *task_out)`: 释放 TaskOutput。
35. `const char *inferlet_task_output_get_result_text(void *task_out)`: 获取结果文本。
36. `const int *inferlet_task_output_get_result_tokens(void *task_out, int *length)`: 获取结果 token 序列。
37. `const int *inferlet_task_output_get_token_offsets(void *task_out)`: 获取各 token 在结果文本中的 UTF-8 字节偏移。
38. `const float *inferlet_task_output_get_token_probability(void *task_out)`: 获取各 token 对应概率。
39. `const int *inferlet_task_output_get_input_tokens(void *task_out, int *length, const char ***token_text)`: 获取输入 token 详情。
40. `const float **inferlet_task_output_get_output_logits(void *task_out, int *length, int *n_vocab)`: 获取输出 logits 矩阵。
41. `const float **inferlet_task_output_get_output_embedding(void *task_out, int *length, int *embed_size)`: 获取输出 embeddings 矩阵。

### 5.5 流式结果读取 (7 个函数)
42. `const char *inferlet_get_stream_text(void *handle, void *task_stream)`: 获取当前流式 chunk 文本（空串表示暂无新词，NULL 表示已完成并自动释放 ticket）。
43. `const int *inferlet_get_stream_tokens(void *task_stream, int *length)`: 获取当前 chunk 的 tokens。
44. `const int *inferlet_get_stream_token_offsets(void *task_stream)`: 获取当前 chunk 的 token 字节偏移。
45. `const float *inferlet_get_stream_token_probability(void *task_stream)`: 获取当前 chunk 的 token 概率。
46. `const float **inferlet_get_stream_output_logits(void *task_stream, int *length, int *n_vocab)`: 获取当前 chunk 的 logits 矩阵。
47. `const float **inferlet_get_stream_output_embedding(void *task_stream, int *length, int *embed_size)`: 获取当前 chunk 的 embedding 矩阵。
48. `const int *inferlet_get_stream_input_tokens(void *task_stream, int *length, const char ***token_text)`: 首次有效 chunk 时获取输入 token 详情。

### 5.6 模型参数构建 `Parameter` (14 个函数)
49. `void *inferlet_parameter_allocate(void)`: 分配 Parameter。
50. `void inferlet_parameter_deallocate(void *param)`: 释放 Parameter。
51. `void inferlet_parameter_set_tokenizer_only(void *param, int value)`: 设置仅 tokenizer 模式。
52. `void inferlet_parameter_set_tokenizer_slow(void *param, int value)`: 设置 slow tokenizer 模式。
53. `void inferlet_parameter_set_vision_max_patch_cnt(void *param, int max_cnt)`: 设置视觉单次最大 patch 数。
54. `void inferlet_parameter_set_vision_max_grid_size(void *param, int max_size)`: 设置视觉最大网格大小。
55. `void inferlet_parameter_set_vision_trans_skipped(void *param, int value)`: 设置是否跳过视觉编码。
56. `void inferlet_parameter_set_decoder_max_token_cnt(void *param, int max_cnt)`: 设置解码单次批大小上限。
57. `void inferlet_parameter_set_device_id(void *param, int device_id)`: 设置指定单设备 ID。
58. `void inferlet_parameter_set_device_list(void *param, int *device_list, int device_cnt)`: 设置多设备列表。
59. `void inferlet_parameter_set_backend_cnt(void *param, int backend_cnt)`: 设置并发 backend 引擎数量。
60. `void inferlet_parameter_set_low_rank_adapter(void *param, int lora_id, const char *file_name)`: 添加 LoRA 权重文件映射。
61. `void inferlet_parameter_set_load_from_file_sync(void *param, int mode)`: 设置加载模式（1 为同步，0 为异步）。
62. `void inferlet_parameter_set_run_config_file(void *param, const char *file_name)`: 设置 JSON 运行配置文件路径。

### 5.7 Tokenizer 功能函数 (2 个函数)
63. `int inferlet_tokenizer_encode(void *handle, const char *text, int text_length, int *tokens, int max_token_length, int *real_token_length, int add_special_token, int parse_special)`: 文本编码为 tokens。
64. `int inferlet_tokenizer_decode(void *handle, const int *tokens, int token_length, char *text, int max_text_length, int *real_text_length, int skip_special_tokens)`: tokens 解码为文本。token ID 必须位于模型词表范围内；越界返回 `INFERLET_E_INVALID_ARGUMENT`，`*real_text_length` 为 0，输出缓冲区保持不变，包括长度查询模式。

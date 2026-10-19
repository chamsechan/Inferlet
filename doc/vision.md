# 图像推理集成

当前 CPU 静态库已接通固定版本 llama.cpp 的 libmtmd。公共 C ABI 不变，仍只需 `include/inferlet.h` 和 `lib/libinferlet.a`，无需额外链接 libmtmd、llama 或 ggml。

## 模型与配置

需要视觉语言模型 GGUF 和与它配套的视觉投影文件（通常名为 `mmproj-*.gguf`）。普通文本模型不会因为传入图像而获得视觉能力。可用模型及模板还受项目固定的 llama.cpp 版本限制；本次实际验证的是 SmolVLM-256M-Instruct Q8_0，不能据此保证所有视觉模型兼容。

将以下内容保存为 `vision.json`，通过 `inferlet_parameter_set_run_config_file` 传入：

```json
{
  "schema_version": 1,
  "model": { "context_size": 4096, "threads": 4 },
  "vision": { "mmproj": "models/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf" }
}
```

`mmproj` 的相对路径以 JSON 所在目录为基准。使用 `inferlet_parameter_set_load_from_file_sync(param, 1)` 可在 `inferlet_load_from_file` 返回时确认模型和投影器都已成功加载；失败返回 NULL。未配置 `vision` 时不加载投影器，原有文本调用不变。

## C 接口示例

下面演示一张 256×256 纯红图像。SmolVLM 投影器的 `clip.vision.patch_size` 为 16，其他模型必须使用自己投影文件中的值。

```c
#include "inferlet.h"
#include <stdlib.h>
#include <string.h>

void *param = inferlet_parameter_allocate();
inferlet_parameter_set_run_config_file(param, "vision.json");
inferlet_parameter_set_load_from_file_sync(param, 1);
void *model = inferlet_load_from_file("models/SmolVLM-256M-Instruct-Q8_0.gguf", param);
inferlet_parameter_deallocate(param);
/* 检查 model != NULL 后继续。 */

unsigned char rgb_chw[3 * 256 * 256] = {0};
memset(rgb_chw, 255, 256 * 256); /* R 平面；G、B 平面为 0。 */
inferlet_multimodal_chat_content content[2] = {0};
content[0].type = "image";
content[0].data = rgb_chw;
content[0].size = sizeof(rgb_chw);
content[0].data_type = 4;
content[0].grid_t = 1;
content[0].grid_h = content[0].grid_w = 256 / 16;
content[1].type = "text";
content[1].data = "What color is this image?";
inferlet_multimodal_chat_history chat = {0};
chat.role = "user";
chat.content = content;
chat.length = 2;
void *input = inferlet_task_input_allocate();
int rc = inferlet_task_input_set_multimodal_chat_history(input, &chat, 1);
/* 检查 rc == INFERLET_OK；setter 返回后可释放或修改 rgb_chw。 */
inferlet_task_input_set_temperature(input, 0);
inferlet_task_input_set_max_output_tokens(input, 32);
void *output = NULL;
rc = inferlet_run(model, input, &output);
if (rc == INFERLET_OK) {
    const char *answer = inferlet_task_output_get_result_text(output);
    /* 在释放 output 前使用 answer。 */
    (void) answer;
    inferlet_task_output_deallocate(output);
}
inferlet_task_input_deallocate(input);
inferlet_unload(model);
```

应用仍需在这些调用前后配对调用 `inferlet_init()` / `inferlet_deinit()`。完整可执行 C 测试见 `tests/integration/test_vision.c`。

JPEG/PNG 需由调用方解码并转换为 CHW 平面 RGB：连续存放整个 R 平面、G 平面、B 平面。高度和宽度应为 patch_size 的整数倍；必要时由调用方缩放或补边。`size` 必须恰好等于 `3 * grid_h * grid_w * patch_size * patch_size`。视觉归一化、模型所需缩放、切图和投影由 libmtmd 处理。

## 支持范围与边界

- 同步、异步、流式共用真实图像编码和生成路径。图像和文本按 content 顺序进入模型，聊天角色和历史通过模型模板保留，最后一条消息必须为 user。
- 可在内容中交错传入多张图像。`cache_key` 非空且 `cache_times >= 2` 时缓存投影后的特征；同一存活 key 的图像内容和网格必须一致，否则返回 `INFERLET_E_INVALID_ARGUMENT`。缓存按使用次数和 `limits.max_visual_cache_bytes` 淘汰。
- `VisualEncoder` 只接受空值或 `"default"`。`VisionMaxPatchCnt`、`VisionMaxGridSize` 保持默认 0，预处理遵守投影器自身规则；自定义限制目前明确拒绝。`VisionTransSkipped=1`、视频、legacy feature、float/half/bf16 特征输入、非默认帧/切片扩展、图像 `beam_size > 1` 返回不支持。
- 当前视觉编码使用 CPU。编码器的共享输出缓冲区受锁保护；文本生成仍按 backend 配置调度。取消为协作式，正在执行的单次视觉编码会在返回后观察取消标志。
- 图像消耗上下文容量。采用非因果图像注意力的模型需要 `model.n_batch` 和 `model.n_ubatch` 能容纳单个图像块的视觉 token；过小会明确报错，不能拆批后悄悄改变注意力含义。M-RoPE 位置处理使用上游 helper；尚未逐模型验证。
- 输入 verbose 将视觉位置表示为 `token=-1`、文本 `"<image>"`，不可将这些位置送入文本 tokenizer decode。流式失败与 EOF 都通过 `inferlet_get_stream_text == NULL` 表示，这是既有 ABI 限制。

## 构建与测试

新增的编译目标只有 `mtmd` 及其静态依赖 `vendor-hash`；工具、common、server、示例程序等上游目标仍关闭，`MTMD_VIDEO=OFF`。上游 libmtmd 的模型注册表也引用音频架构实现，按其库目标编译不代表本项目提供音频接口；没有为了裁剪这些源文件改动 vendor。

测试模型来源为 `ggml-org/SmolVLM-256M-Instruct-GGUF`，固定修订 `b9e4379657e1450d04d02eec8e345667265b0a00`：

- `SmolVLM-256M-Instruct-Q8_0.gguf`
- `mmproj-SmolVLM-256M-Instruct-Q8_0.gguf`

```sh
INFERLET_TEST_VISION_MODEL="$PWD/build_vision/models/SmolVLM-256M-Instruct-Q8_0.gguf" \
INFERLET_TEST_VISION_CONFIG="$PWD/build_vision/vision.json" \
ctest --test-dir build_verify -R integration_vision --output-on-failure
```

未设置这两个环境变量时，视觉测试报告 skipped，不算实际图像验证。

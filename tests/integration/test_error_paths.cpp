#include "inferlet.h"
#include "core/inference_task.h"
#include "support/last_error.h"
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>

using namespace inferlet;

static void check_execution_error(void *model, void *input, int expected) {
    void *output = nullptr;
    assert(inferlet_run(model, input, &output) == expected);
    assert(output == nullptr);
    assert(get_last_error_code() == expected);

    void *task = nullptr;
    assert(inferlet_run_async(model, input, &task) == INFERLET_OK);
    assert(inferlet_wait_task(model, task, &output) == expected);
    assert(output == nullptr);

    assert(inferlet_run_async(model, input, &task) == INFERLET_OK);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    int status;
    while ((status = inferlet_query_task(model, task, &output)) == INFERLET_OK) {
        assert(output == nullptr && "Failed task must not return a successful result");
        assert(std::chrono::steady_clock::now() < deadline);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    assert(status == expected);
    assert(output == nullptr);

    assert(inferlet_task_input_set_stream_block_mode(input, 1) == INFERLET_OK);
    assert(inferlet_run_stream(model, input, &task) == INFERLET_OK);
    auto core = static_cast<StreamTicketInternal *>(task)->core;
    {
        std::unique_lock<std::mutex> lock(core->mutex);
        assert(core->cv.wait_for(lock, std::chrono::seconds(60), [&] { return core->is_terminal(); }));
        assert(core->state == TaskState::Failed);
        assert(core->terminal_status.code == expected);
    }
    // Only empty/incomplete text or pre-generation failures are used here.
    assert(inferlet_get_stream_text(model, task) == nullptr);
    assert(get_last_error_code() == expected);
    assert(*get_last_error() != '\0');
}

int main() {
    const char *path = std::getenv("INFERLET_TEST_MODEL");
    if (!path || !*path) path = "/home/ubuntu/project/llm-ops-agy/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    inferlet_init();
    void *param = inferlet_parameter_allocate();
    assert(param);
    inferlet_parameter_set_load_from_file_sync(param, 1);
    void *model = inferlet_load_from_file(path, param);
    assert(model);
    inferlet_parameter_deallocate(param);

    void *input = inferlet_task_input_allocate();
    assert(input);
    assert(inferlet_task_input_set_prompt_text(input, "Answer:") == INFERLET_OK);
    assert(inferlet_task_input_set_max_output_tokens(input, 1) == INFERLET_OK);
    assert(inferlet_task_input_set_temperature(input, 0) == INFERLET_OK);

    assert(inferlet_task_input_set_low_rank_adapter_id(input, 999) == INFERLET_OK);
    check_execution_error(model, input, INFERLET_E_UNSUPPORTED);
    assert(inferlet_task_input_set_low_rank_adapter_id(input, 0) == INFERLET_OK);

    // Qwen's byte fallback splits this code point; a one-token limit leaves
    // incomplete UTF-8. Both generation engines must report a failed task.
    const char *choices[] = {u8"\U0010FFFF"};
    assert(inferlet_task_input_set_guided_choices(input, choices, 1) == INFERLET_OK);
    for (int beam : {1, 2}) {
        assert(inferlet_task_input_set_beam_size(input, beam) == INFERLET_OK);
        check_execution_error(model, input, INFERLET_E_INFERENCE);
    }

    // A later successful EOF must clear the preceding error.
    assert(inferlet_task_input_set_beam_size(input, 1) == INFERLET_OK);
    choices[0] = "OK";
    assert(inferlet_task_input_set_guided_choices(input, choices, 1) == INFERLET_OK);
    assert(inferlet_task_input_set_max_output_tokens(input, 8) == INFERLET_OK);
    void *stream = nullptr;
    assert(inferlet_run_stream(model, input, &stream) == INFERLET_OK);
    std::string text;
    while (const char *chunk = inferlet_get_stream_text(model, stream)) text += chunk;
    assert(get_last_error_code() == INFERLET_OK);
    assert(text == "OK");

    inferlet_task_input_deallocate(input);
    inferlet_unload(model);
    inferlet_deinit();
    std::cout << "Error propagation across sync/async/query/stream and beam search passed.\n";
}

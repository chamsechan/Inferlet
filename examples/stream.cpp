#include "inferlet.h"
#include <iostream>
#include <string>
#include <cstring>
#include <cstdlib>

static const char *get_default_model() {
    const char *env = std::getenv("INFERLET_TEST_MODEL");
    if (env && std::strlen(env) > 0) {
        return env;
    }
    return "/home/ubuntu/project/llm-ops-agy/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
}

int main(int argc, char **argv) {
    const char *model_path = get_default_model();
    const char *prompt = "Count numbers from 1 to 5 with commas.";

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (std::strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        }
    }

    std::cout << "[example_stream] Initializing Inferlet...\n";
    inferlet_init();

    void *param = inferlet_parameter_allocate();
    inferlet_parameter_set_load_from_file_sync(param, 1);

    std::cout << "[example_stream] Loading model: " << model_path << "\n";
    void *handle = inferlet_load_from_file(model_path, param);
    inferlet_parameter_deallocate(param);

    if (!handle) {
        std::cerr << "[example_stream] Failed to load model: " << model_path << "\n";
        inferlet_deinit();
        return 0;
    }

    void *task_in = inferlet_task_input_allocate();
    inferlet_task_input_set_prompt_text(task_in, prompt);
    inferlet_task_input_set_max_output_tokens(task_in, 32);
    // Enable blocking mode so GetStreamText waits for the next chunk
    inferlet_task_input_set_stream_block_mode(task_in, 1);

    std::cout << "[example_stream] Submitting stream task...\n";
    void *stream_task = nullptr;
    int ret = inferlet_run_stream(handle, task_in, &stream_task);
    inferlet_task_input_deallocate(task_in);

    if (ret != INFERLET_OK || !stream_task) {
        std::cerr << "[example_stream] RunStream failed: " << ret << "\n";
        inferlet_unload(handle);
        inferlet_deinit();
        return 1;
    }

    std::cout << "\n--- Streaming Output ---\n";
    while (true) {
        const char *chunk = inferlet_get_stream_text(handle, stream_task);
        if (!chunk) {
            // NULL indicates the stream is finished and stream_task is automatically deallocated
            break;
        }
        std::cout << chunk << std::flush;
    }

    std::cout << "\n------------------------\n";

    inferlet_unload(handle);
    inferlet_deinit();
    std::cout << "[example_stream] Finished successfully.\n";
    return 0;
}

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
    const char *prompt = "Explain in one sentence why the sky is blue.";

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (std::strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        }
    }

    std::cout << "[example_async] Initializing Inferlet...\n";
    inferlet_init();

    void *param = inferlet_parameter_allocate();
    inferlet_parameter_set_load_from_file_sync(param, 1);

    std::cout << "[example_async] Loading model: " << model_path << "\n";
    void *handle = inferlet_load_from_file(model_path, param);
    inferlet_parameter_deallocate(param);

    if (!handle) {
        std::cerr << "[example_async] Failed to load model: " << model_path << "\n";
        inferlet_deinit();
        return 0;
    }

    void *task_in = inferlet_task_input_allocate();
    inferlet_task_input_set_prompt_text(task_in, prompt);
    inferlet_task_input_set_max_output_tokens(task_in, 32);

    std::cout << "[example_async] Submitting asynchronous task...\n";
    void *task = nullptr;
    int ret = inferlet_run_async(handle, task_in, &task);

    // Deep copy snapshot semantics: task_in can be freed immediately after RunASync!
    inferlet_task_input_deallocate(task_in);

    if (ret != INFERLET_OK || !task) {
        std::cerr << "[example_async] RunASync failed with code: " << ret << "\n";
        inferlet_unload(handle);
        inferlet_deinit();
        return 1;
    }

    std::cout << "[example_async] Waiting for task completion...\n";
    void *task_out = nullptr;
    ret = inferlet_wait_task(handle, task, &task_out);

    if (ret == INFERLET_OK && task_out) {
        const char *result_text = inferlet_task_output_get_result_text(task_out);
        std::cout << "\n--- Async Task Result ---\n";
        std::cout << (result_text ? result_text : "(empty)") << "\n";
        std::cout << "-------------------------\n";
        inferlet_task_output_deallocate(task_out);
    } else {
        std::cerr << "[example_async] Task failed or was cancelled: " << ret << "\n";
    }

    inferlet_unload(handle);
    inferlet_deinit();
    std::cout << "[example_async] Finished successfully.\n";
    return 0;
}

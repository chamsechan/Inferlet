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
    const char *prompt = "Hello! Write a short one-line greeting.";

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (std::strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        }
    }

    std::cout << "[example_sync] Initializing Inferlet...\n";
    inferlet_init();

    void *param = inferlet_parameter_allocate();
    inferlet_parameter_set_load_from_file_sync(param, 1);

    std::cout << "[example_sync] Loading model from: " << model_path << "\n";
    void *handle = inferlet_load_from_file(model_path, param);
    inferlet_parameter_deallocate(param);

    if (!handle) {
        std::cerr << "[example_sync] Failed to load model (file may not exist: " << model_path << ")\n";
        inferlet_deinit();
        return 0;
    }

    std::cout << "[example_sync] Model loaded successfully. Submitting synchronous task...\n";
    std::cout << "[example_sync] Prompt: \"" << prompt << "\"\n";

    void *task_in = inferlet_task_input_allocate();
    inferlet_task_input_set_prompt_text(task_in, prompt);
    inferlet_task_input_set_max_output_tokens(task_in, 32);
    inferlet_task_input_set_temperature(task_in, 0.7f);

    void *task_out = nullptr;
    int ret = inferlet_run(handle, task_in, &task_out);

    if (ret == INFERLET_OK && task_out) {
        const char *result_text = inferlet_task_output_get_result_text(task_out);
        int n_tokens = 0;
        const int *tokens = inferlet_task_output_get_result_tokens(task_out, &n_tokens);
        std::cout << "\n--- Generated Output (" << n_tokens << " tokens) ---\n";
        std::cout << (result_text ? result_text : "(empty)") << "\n";
        std::cout << "------------------------------------\n";
        inferlet_task_output_deallocate(task_out);
    } else {
        std::cerr << "[example_sync] Run failed with error code: " << ret << "\n";
    }

    inferlet_task_input_deallocate(task_in);
    inferlet_unload(handle);
    inferlet_deinit();
    std::cout << "[example_sync] Finished successfully.\n";
    return 0;
}

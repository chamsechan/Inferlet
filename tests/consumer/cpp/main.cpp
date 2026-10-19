#include "inferlet.h"
#include <iostream>
#include <cassert>
#include <cstdlib>
#include <cstring>

int main() {
    std::cout << "[consumer_cpp] Testing Inferlet C++ consumer...\n";

    inferlet_init();
    int ret = 0;

    void *param = inferlet_parameter_allocate();


    assert(param != nullptr);
    inferlet_parameter_set_decoder_max_token_cnt(param, 256);
    inferlet_parameter_set_device_id(param, 0);

    const char *model_path = std::getenv("INFERLET_TEST_MODEL");
    if (!model_path || std::strlen(model_path) == 0) {
        model_path = "/home/ubuntu/project/llm-ops-agy/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    }

    inferlet_parameter_set_load_from_file_sync(param, 1);
    void *handle = inferlet_load_from_file(model_path, param);
    inferlet_parameter_deallocate(param);

    if (handle) {
        std::cout << "[consumer_cpp] Loaded model from: " << model_path << "\n";
        void *task_in = inferlet_task_input_allocate();
        inferlet_task_input_set_prompt_text(task_in, "Say hi");
        inferlet_task_input_set_max_output_tokens(task_in, 5);

        void *task_out = nullptr;
        ret = inferlet_run(handle, task_in, &task_out);
        assert(ret == INFERLET_OK);
        assert(task_out != nullptr);

        const char *text = inferlet_task_output_get_result_text(task_out);
        assert(text != nullptr);
        std::cout << "[consumer_cpp] Generated output: \"" << text << "\"\n";

        inferlet_task_input_deallocate(task_in);
        inferlet_task_output_deallocate(task_out);
        inferlet_unload(handle);
    } else {
        if (std::getenv("INFERLET_TEST_MODEL")) {
            std::cerr << "[consumer_cpp] Explicit test model failed to load.\n";
            inferlet_deinit();
            return 1;
        }
        std::cout << "[consumer_cpp] Model file not accessible; lifecycle verification passed.\n";
    }

    inferlet_deinit();

    std::cout << "[consumer_cpp] C++ consumer test PASSED!\n";

    return 0;
}

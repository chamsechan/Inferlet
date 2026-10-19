#include "inferlet.h"
#include <cassert>
#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <chrono>
#include <thread>
#include "core/model_handle.h"

static void test_sync_execution_limit(void *handle) {
    auto *internal = inferlet::checked_cast<inferlet::ModelHandleInternal>(
        handle, inferlet::ObjectKind::ModelHandle);
    assert(internal && internal->wait_for_ready().is_ok());
    auto model = internal->shared_model;
    void *input = inferlet_task_input_allocate();
    assert(inferlet_task_input_set_prompt_text(input, "Hello") == INFERLET_OK);
    assert(inferlet_task_input_set_max_output_tokens(input, 2) == INFERLET_OK);

    // Occupy the sole backend slot, as a running asynchronous task would.
    model->exec_semaphore->acquire();
    std::atomic<bool> done{false};
    int result = INFERLET_E_INTERNAL;
    void *output = nullptr;
    std::thread worker([&] {
        result = inferlet_run(handle, input, &output);
        done = true;
    });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!done && model->queued == 0 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    bool waited_for_slot = model->queued == 1 && model->running == 0 && !done;
    model->exec_semaphore->release();
    worker.join();
    assert(waited_for_slot);
    assert(result == INFERLET_OK && output != nullptr);
    assert(model->queued == 0 && model->running == 0);
    assert(model->exec_semaphore->count() == 1);
    inferlet_task_output_deallocate(output);
    inferlet_task_input_deallocate(input);
}

static const char *get_model_path() {
    const char *env = std::getenv("INFERLET_TEST_MODEL");
    if (env && std::strlen(env) > 0) {
        return env;
    }
    return "/home/ubuntu/project/llm-ops-agy/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
}

static void test_basic_sync_generation(void *handle) {
    std::cout << "Running test_basic_sync_generation...\n";
    void *task_in = inferlet_task_input_allocate();
    assert(task_in != nullptr);

    int ret = inferlet_task_input_set_prompt_text(task_in, "The capital of France is");
    assert(ret == INFERLET_OK);

    ret = inferlet_task_input_set_max_output_tokens(task_in, 8);
    assert(ret == INFERLET_OK);

    ret = inferlet_task_input_set_temperature(task_in, 0.0f); // greedy
    assert(ret == INFERLET_OK);

    void *task_out = nullptr;
    ret = inferlet_run(handle, task_in, &task_out);
    assert(ret == INFERLET_OK);
    assert(task_out != nullptr);

    const char *text = inferlet_task_output_get_result_text(task_out);
    assert(text != nullptr);
    assert(std::strlen(text) > 0);
    std::cout << "Generated text: \"" << text << "\"\n";

    int token_len = 0;
    const int *tokens = inferlet_task_output_get_result_tokens(task_out, &token_len);
    assert(tokens != nullptr);
    assert(token_len > 0);
    assert(token_len <= 8);

    const int *offsets = inferlet_task_output_get_token_offsets(task_out);
    assert(offsets != nullptr);
    assert(offsets[0] == 0);
    for (int i = 1; i < token_len; ++i) {
        assert(offsets[i] >= offsets[i - 1]);
        assert(offsets[i] <= static_cast<int>(std::strlen(text)));
    }

    // Detokenize and verify text match
    char detok_buf[512] = {0};
    int detok_len = 0;
    ret = inferlet_tokenizer_decode(handle, tokens, token_len, detok_buf, sizeof(detok_buf), &detok_len, 0);
    assert(ret == INFERLET_OK);
    assert(std::string(text) == std::string(detok_buf));

    // Probability type was 0, so GetTokenProbability must return nullptr
    const float *probs = inferlet_task_output_get_token_probability(task_out);
    assert(probs == nullptr);

    inferlet_task_input_deallocate(task_in);
    inferlet_task_output_deallocate(task_out);
    std::cout << "[PASS] test_basic_sync_generation\n";
}

static void test_token_probability(void *handle) {
    std::cout << "Running test_token_probability...\n";

    // Test prob_type = 1 (before sampling)
    {
        void *task_in = inferlet_task_input_allocate();
        assert(task_in != nullptr);

        inferlet_task_input_set_prompt_text(task_in, "One, two,");
        inferlet_task_input_set_max_output_tokens(task_in, 5);
        inferlet_task_input_set_probability_type(task_in, 1);

        void *task_out = nullptr;
        int ret = inferlet_run(handle, task_in, &task_out);
        assert(ret == INFERLET_OK);
        assert(task_out != nullptr);

        int token_len = 0;
        const int *tokens = inferlet_task_output_get_result_tokens(task_out, &token_len);
        assert(tokens != nullptr && token_len > 0);

        const float *probs = inferlet_task_output_get_token_probability(task_out);
        assert(probs != nullptr);
        for (int i = 0; i < token_len; ++i) {
            assert(probs[i] >= 0.0f && probs[i] <= 1.0f);
        }

        inferlet_task_input_deallocate(task_in);
        inferlet_task_output_deallocate(task_out);
    }

    // Test prob_type = 2 (after sampling)
    {
        void *task_in = inferlet_task_input_allocate();
        assert(task_in != nullptr);

        inferlet_task_input_set_prompt_text(task_in, "One, two,");
        inferlet_task_input_set_max_output_tokens(task_in, 5);
        inferlet_task_input_set_probability_type(task_in, 2);

        void *task_out = nullptr;
        int ret = inferlet_run(handle, task_in, &task_out);
        assert(ret == INFERLET_OK);
        assert(task_out != nullptr);

        int token_len = 0;
        const int *tokens = inferlet_task_output_get_result_tokens(task_out, &token_len);
        assert(tokens != nullptr && token_len > 0);

        const float *probs = inferlet_task_output_get_token_probability(task_out);
        assert(probs != nullptr);
        for (int i = 0; i < token_len; ++i) {
            assert(probs[i] >= 0.0f && probs[i] <= 1.0f);
        }

        inferlet_task_input_deallocate(task_in);
        inferlet_task_output_deallocate(task_out);
    }

    std::cout << "[PASS] test_token_probability\n";
}

static void test_max_tokens_exact(void *handle) {
    std::cout << "Running test_max_tokens_exact...\n";
    void *task_in = inferlet_task_input_allocate();
    assert(task_in != nullptr);

    inferlet_task_input_set_prompt_text(task_in, "Once upon a time in a faraway galaxy, there lived a curious robot who");
    inferlet_task_input_set_max_output_tokens(task_in, 4);

    void *task_out = nullptr;
    int ret = inferlet_run(handle, task_in, &task_out);
    assert(ret == INFERLET_OK);
    assert(task_out != nullptr);

    int token_len = 0;
    inferlet_task_output_get_result_tokens(task_out, &token_len);
    assert(token_len == 4);

    inferlet_task_input_deallocate(task_in);
    inferlet_task_output_deallocate(task_out);
    std::cout << "[PASS] test_max_tokens_exact\n";
}

static void test_chat_input_generation(void *handle) {
    std::cout << "Running test_chat_input_generation...\n";
    void *task_in = inferlet_task_input_allocate();
    assert(task_in != nullptr);

    inferlet_chat_history hist[1];
    hist[0].system = nullptr;
    hist[0].query = "Respond with only the number 42.";
    hist[0].response = nullptr;

    int ret = inferlet_task_input_set_chat_history(task_in, hist, 1);
    assert(ret == INFERLET_OK);

    ret = inferlet_task_input_set_max_output_tokens(task_in, 10);
    assert(ret == INFERLET_OK);

    void *task_out = nullptr;
    ret = inferlet_run(handle, task_in, &task_out);
    assert(ret == INFERLET_OK);
    assert(task_out != nullptr);

    const char *text = inferlet_task_output_get_result_text(task_out);
    assert(text != nullptr);
    assert(std::strlen(text) > 0);
    std::cout << "Chat response: \"" << text << "\"\n";

    inferlet_task_input_deallocate(task_in);
    inferlet_task_output_deallocate(task_out);
    std::cout << "[PASS] test_chat_input_generation\n";
}

static void test_prompt_tokens_input_generation(void *handle) {
    std::cout << "Running test_prompt_tokens_input_generation...\n";

    // First encode text to tokens
    const char *raw_prompt = "2 + 2 =";
    int encoded_tokens[64];
    int encoded_count = 0;
    int ret = inferlet_tokenizer_encode(handle, raw_prompt, -1, encoded_tokens, 64, &encoded_count, 1, 0);
    assert(ret == INFERLET_OK);
    assert(encoded_count > 0);

    void *task_in = inferlet_task_input_allocate();
    assert(task_in != nullptr);

    ret = inferlet_task_input_set_prompt_tokens(task_in, encoded_tokens, encoded_count);
    assert(ret == INFERLET_OK);

    ret = inferlet_task_input_set_max_output_tokens(task_in, 6);
    assert(ret == INFERLET_OK);

    void *task_out = nullptr;
    ret = inferlet_run(handle, task_in, &task_out);
    assert(ret == INFERLET_OK);
    assert(task_out != nullptr);

    const char *text = inferlet_task_output_get_result_text(task_out);
    assert(text != nullptr);
    assert(std::strlen(text) > 0);
    std::cout << "Tokens prompt response: \"" << text << "\"\n";

    inferlet_task_input_deallocate(task_in);
    inferlet_task_output_deallocate(task_out);
    std::cout << "[PASS] test_prompt_tokens_input_generation\n";
}

static void test_api_robustness_and_errors(void *handle) {
    std::cout << "Running test_api_robustness_and_errors...\n";

    // API-001: Deallocate(NULL) does not crash
    inferlet_task_output_deallocate(nullptr);
    inferlet_task_input_deallocate(nullptr);

    // API-002: Passing Parameter as TaskInput returns invalid handle
    void *param = inferlet_parameter_allocate();
    void *task_out = nullptr;
    int ret = inferlet_run(handle, param, &task_out);
    assert(ret == INFERLET_E_INVALID_HANDLE);
    assert(task_out == nullptr);
    inferlet_parameter_deallocate(param);

    // NULL handle or input
    ret = inferlet_run(nullptr, nullptr, &task_out);
    assert(ret < 0);

    // Getters with NULL
    assert(inferlet_task_output_get_result_text(nullptr) == nullptr);
    int len = -1;
    assert(inferlet_task_output_get_result_tokens(nullptr, &len) == nullptr);
    assert(len == 0);
    assert(inferlet_task_output_get_token_offsets(nullptr) == nullptr);
    assert(inferlet_task_output_get_token_probability(nullptr) == nullptr);

    // Context overflow test with huge prompt tokens
    void *task_in = inferlet_task_input_allocate();
    std::vector<int> huge_tokens(5000, 100); // 5000 tokens exceeds 4096 context
    ret = inferlet_task_input_set_prompt_tokens(task_in, huge_tokens.data(), static_cast<int>(huge_tokens.size()));
    assert(ret == INFERLET_OK);

    ret = inferlet_run(handle, task_in, &task_out);
    assert(ret < 0);
    assert(task_out == nullptr);
    inferlet_task_input_deallocate(task_in);

    std::cout << "[PASS] test_api_robustness_and_errors\n";
}

static void test_output_stability_and_early_input_deallocate(void *handle) {
    std::cout << "Running test_output_stability_and_early_input_deallocate (API-004 & API-005)...\n";

    void *task_in = inferlet_task_input_allocate();
    inferlet_task_input_set_prompt_text(task_in, "Hello!");
    inferlet_task_input_set_max_output_tokens(task_in, 5);

    void *task_out = nullptr;
    int ret = inferlet_run(handle, task_in, &task_out);
    assert(ret == INFERLET_OK);
    assert(task_out != nullptr);

    // API-004: Immediately deallocate task_in
    inferlet_task_input_deallocate(task_in);

    // API-005: Output getter pointers remain stable and valid across calls
    const char *text1 = inferlet_task_output_get_result_text(task_out);
    const char *text2 = inferlet_task_output_get_result_text(task_out);
    assert(text1 != nullptr && text1 == text2);

    int len1 = 0, len2 = 0;
    const int *tok1 = inferlet_task_output_get_result_tokens(task_out, &len1);
    const int *tok2 = inferlet_task_output_get_result_tokens(task_out, &len2);
    assert(tok1 != nullptr && tok1 == tok2);
    assert(len1 == len2);

    const int *off1 = inferlet_task_output_get_token_offsets(task_out);
    const int *off2 = inferlet_task_output_get_token_offsets(task_out);
    assert(off1 != nullptr && off1 == off2);

    inferlet_task_output_deallocate(task_out);
    std::cout << "[PASS] test_output_stability_and_early_input_deallocate\n";
}

int main() {
    std::cout << "================ Test Sync Generation ================\n";
    inferlet_init();

    const char *model_path = get_model_path();
    std::cout << "Loading model: " << model_path << "\n";

    void *param = inferlet_parameter_allocate();
    assert(param != nullptr);

    void *handle = inferlet_load_from_file(model_path, param);
    inferlet_parameter_deallocate(param);

    if (!handle) {
        std::cerr << "Failed to load model from " << model_path << " - check if file exists.\n";
        inferlet_deinit();
        return 1;
    }

    test_basic_sync_generation(handle);
    test_sync_execution_limit(handle);
    test_token_probability(handle);
    test_max_tokens_exact(handle);
    test_chat_input_generation(handle);
    test_prompt_tokens_input_generation(handle);
    test_api_robustness_and_errors(handle);
    test_output_stability_and_early_input_deallocate(handle);

    inferlet_unload(handle);
    inferlet_deinit();

    std::cout << "================ All Sync Generation Tests Passed! ================\n";
    return 0;
}

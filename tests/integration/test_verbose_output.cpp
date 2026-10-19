#include "inferlet.h"
#include <cassert>
#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>

static const char *get_model_path() {
    const char *env = std::getenv("INFERLET_TEST_MODEL");
    if (env && std::strlen(env) > 0) {
        return env;
    }
    return "/home/ubuntu/project/llm-ops-agy/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
}

static void test_unenabled_getters_return_null(void *handle) {
    std::cout << "Running test_unenabled_getters_return_null...\n";
    void *task_in = inferlet_task_input_allocate();
    inferlet_task_input_set_prompt_text(task_in, "Hello there!");
    inferlet_task_input_set_max_output_tokens(task_in, 3);
    // verbose flags left as 0

    void *task_out = nullptr;
    int ret = inferlet_run(handle, task_in, &task_out);
    assert(ret == INFERLET_OK);
    assert(task_out != nullptr);

    int in_len = -1;
    const char **in_text = reinterpret_cast<const char **>(0x1234);
    const int *in_tok = inferlet_task_output_get_input_tokens(task_out, &in_len, &in_text);
    assert(in_tok == nullptr);
    assert(in_len == 0);
    assert(in_text == nullptr);

    int logits_len = -1;
    int vocab_size = -1;
    const float **logits = inferlet_task_output_get_output_logits(task_out, &logits_len, &vocab_size);
    assert(logits == nullptr);
    assert(logits_len == 0);
    assert(vocab_size == 0);

    int embd_len = -1;
    int embd_dim = -1;
    const float **embd = inferlet_task_output_get_output_embedding(task_out, &embd_len, &embd_dim);
    assert(embd == nullptr);
    assert(embd_len == 0);
    assert(embd_dim == 0);

    inferlet_task_input_deallocate(task_in);
    inferlet_task_output_deallocate(task_out);
    std::cout << "[PASS] test_unenabled_getters_return_null\n";
}

static void test_verbose_input_tokens(void *handle) {
    std::cout << "Running test_verbose_input_tokens...\n";
    void *task_in = inferlet_task_input_allocate();
    const std::string prompt = "Hello, world!";
    inferlet_task_input_set_prompt_text(task_in, prompt.c_str());
    inferlet_task_input_set_max_output_tokens(task_in, 2);
    inferlet_task_input_set_verbose_data_flag(task_in, INFERLET_VERBOSE_F_INPUT_TOKENS);

    void *task_out = nullptr;
    int ret = inferlet_run(handle, task_in, &task_out);
    assert(ret == INFERLET_OK);
    assert(task_out != nullptr);

    int in_len = 0;
    const char **in_text = nullptr;
    const int *in_tok = inferlet_task_output_get_input_tokens(task_out, &in_len, &in_text);
    assert(in_tok != nullptr);
    assert(in_len > 0);
    assert(in_text != nullptr);

    for (int i = 0; i < in_len; ++i) {
        assert(in_text[i] != nullptr);
        std::cout << "Input token " << i << ": id=" << in_tok[i] << ", piece=\"" << in_text[i] << "\"\n";
    }

    // Detokenize input tokens and verify roundtrip
    char detok[256] = {0};
    int detok_len = 0;
    ret = inferlet_tokenizer_decode(handle, in_tok, in_len, detok, sizeof(detok), &detok_len, 0);
    assert(ret == INFERLET_OK);
    assert(prompt == std::string(detok));

    inferlet_task_input_deallocate(task_in);
    inferlet_task_output_deallocate(task_out);
    std::cout << "[PASS] test_verbose_input_tokens\n";
}

static void test_verbose_output_logits_and_consistency(void *handle) {
    std::cout << "Running test_verbose_output_logits_and_consistency...\n";
    void *task_in = inferlet_task_input_allocate();
    inferlet_task_input_set_prompt_text(task_in, "1, 2, 3,");
    inferlet_task_input_set_max_output_tokens(task_in, 3);
    inferlet_task_input_set_probability_type(task_in, 1); // Softmax probability before sampling
    inferlet_task_input_set_verbose_data_flag(task_in, INFERLET_VERBOSE_F_OUTPUT_LOGITS | INFERLET_VERBOSE_F_INPUT_TOKENS);

    void *task_out = nullptr;
    int ret = inferlet_run(handle, task_in, &task_out);
    assert(ret == INFERLET_OK);
    assert(task_out != nullptr);

    int out_tokens_len = 0;
    const int *out_tokens = inferlet_task_output_get_result_tokens(task_out, &out_tokens_len);
    assert(out_tokens != nullptr && out_tokens_len == 3);

    const float *probs = inferlet_task_output_get_token_probability(task_out);
    assert(probs != nullptr);

    int logits_len = 0;
    int vocab_size = 0;
    const float **logits = inferlet_task_output_get_output_logits(task_out, &logits_len, &vocab_size);
    assert(logits != nullptr);
    assert(logits_len == 3);
    assert(vocab_size > 0);
    std::cout << "Logits matrix shape: [" << logits_len << ", " << vocab_size << "]\n";

    // Verify row pointer stability and non-nullness
    for (int r = 0; r < logits_len; ++r) {
        assert(logits[r] != nullptr);
        // Verify values are valid floats
        assert(!std::isnan(logits[r][0]));
        assert(!std::isnan(logits[r][vocab_size - 1]));

        // Compute softmax on logits[r] and verify probability of the selected token matches
        float max_l = logits[r][0];
        for (int v = 1; v < vocab_size; ++v) {
            if (logits[r][v] > max_l) max_l = logits[r][v];
        }
        double sum_exp = 0.0;
        for (int v = 0; v < vocab_size; ++v) {
            sum_exp += std::exp(logits[r][v] - max_l);
        }
        int sel = out_tokens[r];
        double expected_prob = std::exp(logits[r][sel] - max_l) / sum_exp;
        float actual_prob = probs[r];
        float diff = std::abs(static_cast<float>(expected_prob) - actual_prob);
        assert(diff < 1e-4f);
    }

    inferlet_task_input_deallocate(task_in);
    inferlet_task_output_deallocate(task_out);
    std::cout << "[PASS] test_verbose_output_logits_and_consistency\n";
}

static void test_verbose_output_embedding(void *handle) {
    std::cout << "Running test_verbose_output_embedding...\n";
    void *task_in = inferlet_task_input_allocate();
    inferlet_task_input_set_prompt_text(task_in, "Test embeddings");
    inferlet_task_input_set_max_output_tokens(task_in, 2);
    inferlet_task_input_set_verbose_data_flag(task_in, INFERLET_VERBOSE_F_OUTPUT_EMBEDDING);

    void *task_out = nullptr;
    int ret = inferlet_run(handle, task_in, &task_out);
    assert(ret == INFERLET_OK);
    assert(task_out != nullptr);

    int embd_len = 0;
    int embd_dim = 0;
    const float **embd = inferlet_task_output_get_output_embedding(task_out, &embd_len, &embd_dim);
    assert(embd != nullptr);
    assert(embd_len == 2);
    assert(embd_dim > 0);
    std::cout << "Embedding matrix shape: [" << embd_len << ", " << embd_dim << "]\n";

    for (int r = 0; r < embd_len; ++r) {
        assert(embd[r] != nullptr);
        double norm = 0.0;
        for (int d = 0; d < embd_dim; ++d) {
            assert(!std::isnan(embd[r][d]));
            norm += embd[r][d] * embd[r][d];
        }
        assert(norm > 0.0);
    }

    inferlet_task_input_deallocate(task_in);
    inferlet_task_output_deallocate(task_out);
    std::cout << "[PASS] test_verbose_output_embedding\n";
}

static void test_budget_rejection_before_allocation() {
    std::cout << "Running test_budget_rejection_before_allocation...\n";

    // Create a run_config file with a tiny max_verbose_bytes = 100 bytes
    const std::string config_path = "/tmp/inferlet_tiny_budget.json";
    {
        std::ofstream ofs(config_path);
        ofs << "{\n"
            << "  \"schema_version\": 1,\n"
            << "  \"limits\": {\n"
            << "    \"max_verbose_bytes\": 100\n"
            << "  }\n"
            << "}\n";
    }

    void *param = inferlet_parameter_allocate();
    inferlet_parameter_set_run_config_file(param, config_path.c_str());

    const char *model_path = get_model_path();
    void *handle = inferlet_load_from_file(model_path, param);
    inferlet_parameter_deallocate(param);
    assert(handle != nullptr);

    // Request logits for 5 tokens: 5 * ~151646 * 4 = ~3MB >> 100 bytes
    void *task_in = inferlet_task_input_allocate();
    inferlet_task_input_set_prompt_text(task_in, "Hello");
    inferlet_task_input_set_max_output_tokens(task_in, 5);
    inferlet_task_input_set_verbose_data_flag(task_in, INFERLET_VERBOSE_F_OUTPUT_LOGITS);

    void *task_out = nullptr;
    int ret = inferlet_run(handle, task_in, &task_out);
    // Must be rejected
    assert(ret < 0);
    assert(task_out == nullptr);

    inferlet_task_input_deallocate(task_in);
    inferlet_unload(handle);
    std::remove(config_path.c_str());

    std::cout << "[PASS] test_budget_rejection_before_allocation\n";
}

int main() {
    std::cout << "================ Test Verbose Output ================\n";
    inferlet_init();

    const char *model_path = get_model_path();
    std::cout << "Loading model: " << model_path << "\n";

    void *param = inferlet_parameter_allocate();
    void *handle = inferlet_load_from_file(model_path, param);
    inferlet_parameter_deallocate(param);

    if (!handle) {
        std::cerr << "Failed to load model from " << model_path << "\n";
        inferlet_deinit();
        return 1;
    }

    test_unenabled_getters_return_null(handle);
    test_verbose_input_tokens(handle);
    test_verbose_output_logits_and_consistency(handle);
    test_verbose_output_embedding(handle);

    inferlet_unload(handle);

    // Test budget rejection with custom configuration
    test_budget_rejection_before_allocation();

    inferlet_deinit();
    std::cout << "================ All Verbose Output Tests Passed! ================\n";
    return 0;
}

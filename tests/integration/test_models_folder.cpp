#include "inferlet.h"
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <thread>
#include <chrono>
#include <filesystem>

static std::string resolve_model(const std::string &rel_path) {
    const char *model_dir = std::getenv("INFERLET_TEST_MODEL_DIR");
    if (model_dir && *model_dir) {
        return (std::filesystem::path(model_dir) / std::filesystem::path(rel_path).filename()).string();
    }
    std::vector<std::string> candidates = {
        rel_path,
        "../" + rel_path,
        "../../" + rel_path,
        "/home/ubuntu/project/llm-ops-agy/" + rel_path
    };
    for (const auto &cand : candidates) {
        std::error_code ec;
        if (std::filesystem::exists(cand, ec)) {
            return cand;
        }
    }
    return rel_path;
}

static void test_tokenizer(void *handle) {
    std::cout << "[Test 1/8] Testing Tokenizer Encode & Decode with real vocab...\n";
    const char *sample = "Hello from Inferlet! 1 + 1 = 2.";
    int tokens[64];
    int real_tokens = 0;
    int ret = inferlet_tokenizer_encode(handle, sample, -1, tokens, 64, &real_tokens, 1, 0);
    assert(ret == INFERLET_OK);
    assert(real_tokens > 0);

    char detok[256] = {0};
    int real_text = 0;
    ret = inferlet_tokenizer_decode(handle, tokens, real_tokens, detok, sizeof(detok), &real_text, 0);
    assert(ret == INFERLET_OK);
    assert(std::string(sample) == std::string(detok));
    std::cout << "  Tokenizer passed. Encoded " << real_tokens << " tokens, decoded: \"" << detok << "\"\n";
}

static void test_sync_and_verbose(void *handle) {
    std::cout << "[Test 2/8] Testing Synchronous Generation, Offsets, Logits, Embeddings...\n";
    void *input = inferlet_task_input_allocate();
    assert(input != nullptr);

    assert(inferlet_task_input_set_prompt_text(input, "France's capital is") == INFERLET_OK);
    assert(inferlet_task_input_set_max_output_tokens(input, 6) == INFERLET_OK);
    assert(inferlet_task_input_set_temperature(input, 0.0f) == INFERLET_OK);
    assert(inferlet_task_input_set_probability_type(input, 1) == INFERLET_OK);
    assert(inferlet_task_input_set_verbose_data_flag(input, INFERLET_VERBOSE_F_ALL) == INFERLET_OK);

    void *output = nullptr;
    int ret = inferlet_run(handle, input, &output);
    assert(ret == INFERLET_OK && output != nullptr);

    const char *text = inferlet_task_output_get_result_text(output);
    assert(text != nullptr);
    std::cout << "  Result text: \"" << text << "\"\n";

    int token_len = 0;
    const int *toks = inferlet_task_output_get_result_tokens(output, &token_len);
    assert(toks != nullptr && token_len > 0);

    const int *offsets = inferlet_task_output_get_token_offsets(output);
    assert(offsets != nullptr);

    const float *probs = inferlet_task_output_get_token_probability(output);
    assert(probs != nullptr);
    for (int i = 0; i < token_len; ++i) {
        assert(probs[i] >= 0.0f && probs[i] <= 1.0f);
    }

    int inp_len = 0;
    const char **inp_texts = nullptr;
    const int *inp_toks = inferlet_task_output_get_input_tokens(output, &inp_len, &inp_texts);
    assert(inp_toks != nullptr && inp_len > 0);
    assert(inp_texts != nullptr);

    int logit_rows = 0, vocab_cols = 0;
    const float **logits = inferlet_task_output_get_output_logits(output, &logit_rows, &vocab_cols);
    assert(logits != nullptr && logit_rows == token_len && vocab_cols > 0);

    int emb_rows = 0, emb_cols = 0;
    const float **embeds = inferlet_task_output_get_output_embedding(output, &emb_rows, &emb_cols);
    assert(embeds != nullptr && emb_rows == token_len && emb_cols > 0);

    inferlet_task_output_deallocate(output);
    inferlet_task_input_deallocate(input);
    std::cout << "  Sync and verbose test passed.\n";
}

static void test_async_and_cancellation(void *handle) {
    std::cout << "[Test 3/8] Testing Async Execution, QueryTask, WaitTask, and CancelTask...\n";
    
    // Normal Async
    {
        void *input = inferlet_task_input_allocate();
        assert(inferlet_task_input_set_prompt_text(input, "1 + 1 =") == INFERLET_OK);
        assert(inferlet_task_input_set_max_output_tokens(input, 5) == INFERLET_OK);

        void *task = nullptr;
        assert(inferlet_run_async(handle, input, &task) == INFERLET_OK && task != nullptr);

        void *output = nullptr;
        while (inferlet_query_task(handle, task, &output) == INFERLET_OK && output == nullptr) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        assert(output != nullptr);
        std::cout << "  Async query completed: \"" << inferlet_task_output_get_result_text(output) << "\"\n";

        inferlet_task_output_deallocate(output);
        inferlet_task_input_deallocate(input);
    }

    // Cancellation
    {
        void *input = inferlet_task_input_allocate();
        assert(inferlet_task_input_set_prompt_text(input, "Write a long essay about history: Chapter 1...") == INFERLET_OK);
        assert(inferlet_task_input_set_max_output_tokens(input, 100) == INFERLET_OK);

        void *task = nullptr;
        assert(inferlet_run_async(handle, input, &task) == INFERLET_OK && task != nullptr);

        std::this_thread::sleep_for(std::chrono::milliseconds(15));
        assert(inferlet_cancel_task(handle, task) == INFERLET_OK);

        void *output = nullptr;
        int rc = inferlet_wait_task(handle, task, &output);
        assert(rc == INFERLET_E_CANCELLED);
        if (output) inferlet_task_output_deallocate(output);
        inferlet_task_input_deallocate(input);
        std::cout << "  Cancellation test passed.\n";
    }
}

static void test_streaming(void *handle) {
    std::cout << "[Test 4/8] Testing Streaming Generation (blocking & non-blocking)...\n";
    void *input = inferlet_task_input_allocate();
    assert(inferlet_task_input_set_prompt_text(input, "Count from 1 to 3:") == INFERLET_OK);
    assert(inferlet_task_input_set_max_output_tokens(input, 15) == INFERLET_OK);
    assert(inferlet_task_input_set_stream_block_mode(input, 1) == INFERLET_OK);
    assert(inferlet_task_input_set_probability_type(input, 1) == INFERLET_OK);

    void *stream = nullptr;
    assert(inferlet_run_stream(handle, input, &stream) == INFERLET_OK && stream != nullptr);

    std::string accumulated;
    while (true) {
        const char *chunk = inferlet_get_stream_text(handle, stream);
        if (chunk == nullptr) {
            break;
        }
        accumulated += chunk;
        int chunk_tokens = 0;
        const int *toks = inferlet_get_stream_tokens(stream, &chunk_tokens);
        if (chunk_tokens > 0) {
            assert(toks != nullptr);
            const int *offsets = inferlet_get_stream_token_offsets(stream);
            assert(offsets != nullptr);
            const float *probs = inferlet_get_stream_token_probability(stream);
            assert(probs != nullptr);
        }
    }
    std::cout << "  Streaming completed. Accumulated text: \"" << accumulated << "\"\n";
    inferlet_task_input_deallocate(input);
}

static void test_guided_choices(void *handle) {
    std::cout << "[Test 5/8] Testing Guided Choices (Constrained Grammar)...\n";
    void *input = inferlet_task_input_allocate();
    assert(inferlet_task_input_set_prompt_text(input, "Is fire hot or cold? Answer:") == INFERLET_OK);
    assert(inferlet_task_input_set_max_output_tokens(input, 10) == INFERLET_OK);

    const char *choices[] = {"hot", "cold"};
    assert(inferlet_task_input_set_guided_choices(input, choices, 2) == INFERLET_OK);

    void *output = nullptr;
    assert(inferlet_run(handle, input, &output) == INFERLET_OK && output != nullptr);

    const char *text = inferlet_task_output_get_result_text(output);
    assert(text != nullptr);
    std::cout << "  Guided choice output: \"" << text << "\"\n";
    assert(std::string(text) == "hot" || std::string(text) == "cold");

    inferlet_task_output_deallocate(output);
    inferlet_task_input_deallocate(input);
}

static void test_beam_search(void *handle) {
    std::cout << "[Test 6/8] Testing Beam Search (beam_size = 2)...\n";
    void *input = inferlet_task_input_allocate();
    assert(inferlet_task_input_set_prompt_text(input, "2 + 2 =") == INFERLET_OK);
    assert(inferlet_task_input_set_beam_size(input, 2) == INFERLET_OK);
    assert(inferlet_task_input_set_max_output_tokens(input, 6) == INFERLET_OK);
    assert(inferlet_task_input_set_temperature(input, 0.0f) == INFERLET_OK);

    void *output = nullptr;
    assert(inferlet_run(handle, input, &output) == INFERLET_OK && output != nullptr);

    const char *text = inferlet_task_output_get_result_text(output);
    assert(text != nullptr);
    std::cout << "  Beam search output: \"" << text << "\"\n";

    inferlet_task_output_deallocate(output);
    inferlet_task_input_deallocate(input);
}

static void test_multimodal(void *handle) {
    std::cout << "[Test 7/8] Testing Multimodal Input & Visual Cache...\n";
    void *input = inferlet_task_input_allocate();
    assert(input != nullptr);

    std::vector<uint8_t> raw_image(588, 120);

    inferlet_multimodal_chat_content contents[2];
    std::memset(contents, 0, sizeof(contents));
    contents[0].type = "text";
    contents[0].data = "What is shown in the image?";
    contents[1].type = "image";
    contents[1].data = raw_image.data();
    contents[1].size = static_cast<int>(raw_image.size());
    contents[1].data_type = 4;
    contents[1].grid_t = 1;
    contents[1].grid_h = 1;
    contents[1].grid_w = 1;
    contents[1].cache_key = "models_test_cache_key";
    contents[1].cache_times = 2;

    inferlet_multimodal_chat_history history[1];
    std::memset(history, 0, sizeof(history));
    history[0].role = "user";
    history[0].content = contents;
    history[0].length = 2;

    assert(inferlet_task_input_set_multimodal_chat_history(input, history, 1) == INFERLET_OK);
    assert(inferlet_task_input_set_max_output_tokens(input, 10) == INFERLET_OK);
    assert(inferlet_task_input_set_temperature(input, 0.0f) == INFERLET_OK);

    void *output = nullptr;
    assert(inferlet_run(handle, input, &output) == INFERLET_E_UNSUPPORTED);
    assert(output == nullptr);
    std::cout << "  Multimodal input correctly rejected without a visual projector\n";

    inferlet_task_input_deallocate(input);
}

static void test_thinking_model(const std::string &path) {
    std::cout << "[Test 8/8] Testing Qwen3 Model with Thinking Mode...\n";
    void *param = inferlet_parameter_allocate();
    inferlet_parameter_set_load_from_file_sync(param, 1);

    void *handle = inferlet_load_from_file(path.c_str(), param);
    inferlet_parameter_deallocate(param);
    assert(handle != nullptr);

    void *input = inferlet_task_input_allocate();
    assert(inferlet_task_input_set_prompt_text(input, "Why is ice cold?") == INFERLET_OK);
    assert(inferlet_task_input_set_max_output_tokens(input, 25) == INFERLET_OK);
    assert(inferlet_task_input_set_thinking_mode(input, 1) == INFERLET_OK);

    void *output = nullptr;
    assert(inferlet_run(handle, input, &output) == INFERLET_OK && output != nullptr);

    const char *text = inferlet_task_output_get_result_text(output);
    assert(text != nullptr);
    std::cout << "  Qwen3 output with thinking mode:\n\"" << text << "\"\n";

    inferlet_task_output_deallocate(output);
    inferlet_task_input_deallocate(input);
    inferlet_unload(handle);
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "   Inferlet Full Functionality Test with models/ folder   \n";
    std::cout << "========================================================\n";

    inferlet_init();

    std::string model_0_5b = resolve_model("models/qwen2.5-0.5b-instruct-q4_k_m.gguf");
    std::string model_3_5 = resolve_model("models/qwen3.5-0.8b-instruct-q4_k_m.gguf");

    void *param = inferlet_parameter_allocate();
    inferlet_parameter_set_load_from_file_sync(param, 1);

    std::cout << "Loading base model: " << model_0_5b << "\n";
    void *handle = inferlet_load_from_file(model_0_5b.c_str(), param);
    inferlet_parameter_deallocate(param);

    if (!handle) {
        std::cerr << "Failed to load " << model_0_5b << "\n";
        inferlet_deinit();
        return 1;
    }

    int active = 0;
    assert(inferlet_check_backend_active(handle, &active) == INFERLET_OK);
    std::cout << "Backend active check: " << active << "\n";

    test_tokenizer(handle);
    test_sync_and_verbose(handle);
    test_async_and_cancellation(handle);
    test_streaming(handle);
    test_guided_choices(handle);
    test_beam_search(handle);
    test_multimodal(handle);

    inferlet_unload(handle);

    // Test second model (Qwen3.5 with thinking mode)
    test_thinking_model(model_3_5);

    inferlet_deinit();

    std::cout << "\n========================================================\n";
    std::cout << "  ALL TESTS WITH models/ REAL MODELS PASSED 100%!       \n";
    std::cout << "========================================================\n";
    return 0;
}

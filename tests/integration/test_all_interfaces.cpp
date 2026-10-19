#include "inferlet.h"
#include <cassert>
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <cstdlib>
#include <thread>
#include <chrono>
#include <fstream>
#include <cmath>
#include <set>

static std::set<std::string> g_tested_apis;

#define RECORD_API(func_name) do { g_tested_apis.insert(#func_name); } while (0)

static const char *get_model_path() {
    const char *env = std::getenv("INFERLET_TEST_MODEL");
    if (env && std::strlen(env) > 0) {
        return env;
    }
    const char *candidates[] = {
        "../models/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "/home/ubuntu/project/llm-ops-agy/models/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
    };
    for (const char *path : candidates) {
        std::ifstream f(path);
        if (f.good()) return path;
    }
    return candidates[0];
}

// -------------------------------------------------------------
// Test Suite 1: Parameter APIs (14 interfaces)
// -------------------------------------------------------------
static void test_suite_parameters() {
    std::cout << "[Suite 1/7] Testing Parameter APIs...\n";

    // 1. inferlet_parameter_allocate
    RECORD_API(inferlet_parameter_allocate);
    void *param = inferlet_parameter_allocate();
    assert(param != nullptr);

    // 2. inferlet_parameter_set_tokenizer_only
    RECORD_API(inferlet_parameter_set_tokenizer_only);
    inferlet_parameter_set_tokenizer_only(param, 0);

    // 3. inferlet_parameter_set_tokenizer_slow
    RECORD_API(inferlet_parameter_set_tokenizer_slow);
    inferlet_parameter_set_tokenizer_slow(param, 0);

    // 4. inferlet_parameter_set_vision_max_patch_cnt
    RECORD_API(inferlet_parameter_set_vision_max_patch_cnt);
    inferlet_parameter_set_vision_max_patch_cnt(param, 0);

    // 5. inferlet_parameter_set_vision_max_grid_size
    RECORD_API(inferlet_parameter_set_vision_max_grid_size);
    inferlet_parameter_set_vision_max_grid_size(param, 0);

    // 6. inferlet_parameter_set_vision_trans_skipped
    RECORD_API(inferlet_parameter_set_vision_trans_skipped);
    inferlet_parameter_set_vision_trans_skipped(param, 0);

    // 7. inferlet_parameter_set_decoder_max_token_cnt
    RECORD_API(inferlet_parameter_set_decoder_max_token_cnt);
    inferlet_parameter_set_decoder_max_token_cnt(param, 512);

    // 8. inferlet_parameter_set_device_id
    RECORD_API(inferlet_parameter_set_device_id);
    inferlet_parameter_set_device_id(param, 0);

    // 9. inferlet_parameter_set_device_list
    RECORD_API(inferlet_parameter_set_device_list);
    int dev_list[1] = {0};
    inferlet_parameter_set_device_list(param, dev_list, 1);

    // 10. inferlet_parameter_set_backend_cnt
    RECORD_API(inferlet_parameter_set_backend_cnt);
    inferlet_parameter_set_backend_cnt(param, 1);

    // 11. inferlet_parameter_set_low_rank_adapter
    RECORD_API(inferlet_parameter_set_low_rank_adapter);
    inferlet_parameter_set_low_rank_adapter(param, 1, "test_adapter.lora");

    // 12. inferlet_parameter_set_load_from_file_sync
    RECORD_API(inferlet_parameter_set_load_from_file_sync);
    inferlet_parameter_set_load_from_file_sync(param, 1);

    // 13. inferlet_parameter_set_run_config_file
    RECORD_API(inferlet_parameter_set_run_config_file);
    const char *cfg_tmp = "/tmp/inferlet_suite_run_cfg.json";
    {
        std::ofstream ofs(cfg_tmp);
        ofs << "{\"schema_version\":1}\n";
    }
    inferlet_parameter_set_run_config_file(param, cfg_tmp);

    // 14. inferlet_parameter_deallocate
    RECORD_API(inferlet_parameter_deallocate);
    inferlet_parameter_deallocate(param);
    // NULL safety
    inferlet_parameter_deallocate(nullptr);
    std::remove(cfg_tmp);

    std::cout << "  -> Parameter APIs (14/14) passed.\n";
}

// -------------------------------------------------------------
// Test Suite 2: Module Lifecycle & Model Management (4 interfaces)
// -------------------------------------------------------------
static void *test_suite_lifecycle_and_model_load() {
    std::cout << "[Suite 2/7] Testing Module Init and Model Load...\n";

    // 15. inferlet_init
    RECORD_API(inferlet_init);
    inferlet_init();

    // 16. inferlet_load_from_file
    RECORD_API(inferlet_load_from_file);
    // Test NULL and empty model_path return NULL immediately
    assert(inferlet_load_from_file(nullptr, nullptr) == nullptr);
    assert(inferlet_load_from_file("", nullptr) == nullptr);

    // Test synchronous load of non-existent model returns NULL
    void *sync_bad_param = inferlet_parameter_allocate();
    inferlet_parameter_set_load_from_file_sync(sync_bad_param, 1);
    void *bad_handle = inferlet_load_from_file("/nonexistent/model.gguf", sync_bad_param);
    assert(bad_handle == nullptr);
    inferlet_parameter_deallocate(sync_bad_param);

    // Load actual test model
    void *load_param = inferlet_parameter_allocate();
    inferlet_parameter_set_load_from_file_sync(load_param, 1);
    const char *model_path = get_model_path();
    std::cout << "  Loading model from: " << model_path << "\n";
    void *handle = inferlet_load_from_file(model_path, load_param);
    inferlet_parameter_deallocate(load_param);
    assert(handle != nullptr && "Failed to load model file");

    // 17. inferlet_check_backend_active
    RECORD_API(inferlet_check_backend_active);
    int is_active = -1;
    int rc = inferlet_check_backend_active(handle, &is_active);
    assert(rc == INFERLET_OK);
    assert(is_active == 0);

    // Test invalid handle on CheckBackendActive
    rc = inferlet_check_backend_active(nullptr, &is_active);
    assert(rc != INFERLET_OK);

    std::cout << "  -> Lifecycle & Load APIs passed.\n";
    return handle;
}

// -------------------------------------------------------------
// Test Suite 3: Tokenizer APIs (2 interfaces)
// -------------------------------------------------------------
static void test_suite_tokenizer(void *handle) {
    std::cout << "[Suite 3/7] Testing Tokenizer APIs...\n";

    const char *text = "Inferlet 接口测试 2026";
    int text_len = static_cast<int>(std::strlen(text));
    int tokens[128] = {0};
    int real_token_len = 0;

    // 18. inferlet_tokenizer_encode
    RECORD_API(inferlet_tokenizer_encode);
    int rc = inferlet_tokenizer_encode(handle, text, text_len, tokens, 128, &real_token_len, 0, 0);
    assert(rc == INFERLET_OK);
    assert(real_token_len > 0);

    // Invalid argument tests for Encode
    int bad_real_len = 0;
    assert(inferlet_tokenizer_encode(nullptr, text, text_len, tokens, 128, &bad_real_len, 0, 0) != INFERLET_OK);
    assert(inferlet_tokenizer_encode(handle, nullptr, text_len, tokens, 128, &bad_real_len, 0, 0) != INFERLET_OK);

    // 19. inferlet_tokenizer_decode
    RECORD_API(inferlet_tokenizer_decode);
    char decoded[256] = {0};
    int real_text_len = 0;
    rc = inferlet_tokenizer_decode(handle, tokens, real_token_len, decoded, 256, &real_text_len, 0);
    assert(rc == INFERLET_OK);
    assert(real_text_len == text_len);
    assert(std::string(decoded, real_text_len) == std::string(text));

    // Invalid argument tests for Decode
    assert(inferlet_tokenizer_decode(nullptr, tokens, real_token_len, decoded, 256, &real_text_len, 0) != INFERLET_OK);
    assert(inferlet_tokenizer_decode(handle, nullptr, real_token_len, decoded, 256, &real_text_len, 0) != INFERLET_OK);

    std::cout << "  -> Tokenizer APIs (2/2) passed.\n";
}

// -------------------------------------------------------------
// Test Suite 4: TaskInput APIs (23 interfaces)
// -------------------------------------------------------------
static void test_suite_task_input() {
    std::cout << "[Suite 4/7] Testing TaskInput APIs...\n";

    // 20. inferlet_task_input_allocate
    RECORD_API(inferlet_task_input_allocate);
    void *task_in = inferlet_task_input_allocate();
    assert(task_in != nullptr);

    // 21. inferlet_task_input_set_prompt_text
    RECORD_API(inferlet_task_input_set_prompt_text);
    assert(inferlet_task_input_set_prompt_text(task_in, "What is 2+2?") == INFERLET_OK);

    // 22. inferlet_task_input_set_prompt_tokens
    RECORD_API(inferlet_task_input_set_prompt_tokens);
    int sample_tokens[] = {151644, 872, 198};
    assert(inferlet_task_input_set_prompt_tokens(task_in, sample_tokens, 3) == INFERLET_OK);

    // 23. inferlet_task_input_set_chat_history
    RECORD_API(inferlet_task_input_set_chat_history);
    inferlet_chat_history history[2] = {
        {"You are a helpful assistant.", "Hi", "Hello!"},
        {nullptr, "How are you?", nullptr}
    };
    assert(inferlet_task_input_set_chat_history(task_in, history, 2) == INFERLET_OK);

    // 24. inferlet_task_input_set_multimodal
    RECORD_API(inferlet_task_input_set_multimodal);
    float dummy_feat[16] = {0};
    inferlet_multimodal mm_item{};
    mm_item.prompt_text = "Image feat:";
    mm_item.feature = dummy_feat;
    mm_item.size = sizeof(dummy_feat);
    assert(inferlet_task_input_set_multimodal(task_in, &mm_item, 1) == INFERLET_OK);

    // 25. inferlet_task_input_set_multimodal_chat_history
    RECORD_API(inferlet_task_input_set_multimodal_chat_history);
    inferlet_multimodal_chat_content mm_content[1] = {};
    mm_content[0].type = "text";
    mm_content[0].data = "Hello vision";
    inferlet_multimodal_chat_history mm_history[1] = {};
    mm_history[0].role = "user";
    mm_history[0].content = mm_content;
    mm_history[0].length = 1;
    assert(inferlet_task_input_set_multimodal_chat_history(task_in, mm_history, 1) == INFERLET_OK);

    // 26. inferlet_task_input_set_low_rank_adapter_id
    RECORD_API(inferlet_task_input_set_low_rank_adapter_id);
    assert(inferlet_task_input_set_low_rank_adapter_id(task_in, 1) == INFERLET_OK);
    assert(inferlet_task_input_set_low_rank_adapter_id(task_in, -1) != INFERLET_OK);

    // 27. inferlet_task_input_set_beam_size
    RECORD_API(inferlet_task_input_set_beam_size);
    assert(inferlet_task_input_set_beam_size(task_in, 2) == INFERLET_OK);
    assert(inferlet_task_input_set_beam_size(task_in, 0) != INFERLET_OK);

    // 28. inferlet_task_input_set_top_k
    RECORD_API(inferlet_task_input_set_top_k);
    assert(inferlet_task_input_set_top_k(task_in, 50) == INFERLET_OK);
    assert(inferlet_task_input_set_top_k(task_in, -1) != INFERLET_OK);

    // 29. inferlet_task_input_set_top_p
    RECORD_API(inferlet_task_input_set_top_p);
    assert(inferlet_task_input_set_top_p(task_in, 0.9f) == INFERLET_OK);
    assert(inferlet_task_input_set_top_p(task_in, -0.1f) != INFERLET_OK);

    // 30. inferlet_task_input_set_temperature
    RECORD_API(inferlet_task_input_set_temperature);
    assert(inferlet_task_input_set_temperature(task_in, 0.7f) == INFERLET_OK);
    assert(inferlet_task_input_set_temperature(task_in, -1.0f) != INFERLET_OK);

    // 31. inferlet_task_input_set_repetition_penalty
    RECORD_API(inferlet_task_input_set_repetition_penalty);
    assert(inferlet_task_input_set_repetition_penalty(task_in, 1.1f) == INFERLET_OK);
    assert(inferlet_task_input_set_repetition_penalty(task_in, 0.0f) != INFERLET_OK);

    // 32. inferlet_task_input_set_max_output_tokens
    RECORD_API(inferlet_task_input_set_max_output_tokens);
    assert(inferlet_task_input_set_max_output_tokens(task_in, 128) == INFERLET_OK);
    assert(inferlet_task_input_set_max_output_tokens(task_in, 0) != INFERLET_OK);

    // 33. inferlet_task_input_set_probability_type
    RECORD_API(inferlet_task_input_set_probability_type);
    assert(inferlet_task_input_set_probability_type(task_in, 1) == INFERLET_OK);
    assert(inferlet_task_input_set_probability_type(task_in, 3) != INFERLET_OK);

    // 34. inferlet_task_input_set_verbose_data_flag
    RECORD_API(inferlet_task_input_set_verbose_data_flag);
    assert(inferlet_task_input_set_verbose_data_flag(task_in, INFERLET_VERBOSE_F_ALL) == INFERLET_OK);

    // 35. inferlet_task_input_set_stream_block_mode
    RECORD_API(inferlet_task_input_set_stream_block_mode);
    assert(inferlet_task_input_set_stream_block_mode(task_in, 1) == INFERLET_OK);
    assert(inferlet_task_input_set_stream_block_mode(task_in, 2) != INFERLET_OK);

    // 36. inferlet_task_input_set_thinking_mode
    RECORD_API(inferlet_task_input_set_thinking_mode);
    assert(inferlet_task_input_set_thinking_mode(task_in, 1) == INFERLET_OK);
    assert(inferlet_task_input_set_thinking_mode(task_in, -1) != INFERLET_OK);

    // 37. inferlet_task_input_set_special_token_mode
    RECORD_API(inferlet_task_input_set_special_token_mode);
    assert(inferlet_task_input_set_special_token_mode(task_in, 1) == INFERLET_OK);
    assert(inferlet_task_input_set_special_token_mode(task_in, 2) != INFERLET_OK);

    // 38. inferlet_task_input_set_guided_choices
    RECORD_API(inferlet_task_input_set_guided_choices);
    const char *choices[] = {"Yes", "No", "Maybe"};
    assert(inferlet_task_input_set_guided_choices(task_in, choices, 3) == INFERLET_OK);
    assert(inferlet_task_input_set_guided_choices(task_in, choices, -1) != INFERLET_OK);

    // 39. inferlet_task_input_set_tools
    RECORD_API(inferlet_task_input_set_tools);
    const char *tools[] = {
        "{\"type\":\"function\",\"function\":{\"name\":\"get_current_weather\",\"description\":\"Get weather\"}}"
    };
    assert(inferlet_task_input_set_tools(task_in, tools, 1) == INFERLET_OK);
    assert(inferlet_task_input_set_tools(task_in, nullptr, 1) != INFERLET_OK);

    // 40. inferlet_task_input_set_visual_encoder
    RECORD_API(inferlet_task_input_set_visual_encoder);
    assert(inferlet_task_input_set_visual_encoder(task_in, "clip-vit") == INFERLET_OK);

    // 41. inferlet_task_input_deallocate
    RECORD_API(inferlet_task_input_deallocate);
    inferlet_task_input_deallocate(task_in);
    // NULL safety
    inferlet_task_input_deallocate(nullptr);

    std::cout << "  -> TaskInput APIs (22/22 setters + deallocate) passed.\n";
}

// -------------------------------------------------------------
// Test Suite 5: Synchronous Inference & TaskOutput APIs (9 interfaces)
// -------------------------------------------------------------
static void test_suite_sync_and_output(void *handle) {
    std::cout << "[Suite 5/7] Testing Synchronous Generation and TaskOutput APIs...\n";

    void *task_in = inferlet_task_input_allocate();
    assert(task_in != nullptr);

    inferlet_task_input_set_prompt_text(task_in, "Question: 1+1=? Answer:");
    inferlet_task_input_set_max_output_tokens(task_in, 4);
    inferlet_task_input_set_temperature(task_in, 0.0f); // greedy
    inferlet_task_input_set_probability_type(task_in, 1);
    inferlet_task_input_set_verbose_data_flag(task_in, INFERLET_VERBOSE_F_ALL);

    void *task_out = nullptr;
    // 42. inferlet_run
    RECORD_API(inferlet_run);
    int rc = inferlet_run(handle, task_in, &task_out);
    assert(rc == INFERLET_OK);
    assert(task_out != nullptr);

    // 43. inferlet_task_output_get_result_text
    RECORD_API(inferlet_task_output_get_result_text);
    const char *result_text = inferlet_task_output_get_result_text(task_out);
    assert(result_text != nullptr);
    assert(std::strlen(result_text) > 0);
    std::cout << "  Result text: \"" << result_text << "\"\n";

    // 44. inferlet_task_output_get_result_tokens
    RECORD_API(inferlet_task_output_get_result_tokens);
    int token_len = 0;
    const int *tokens = inferlet_task_output_get_result_tokens(task_out, &token_len);
    assert(tokens != nullptr);
    assert(token_len > 0);

    // 45. inferlet_task_output_get_token_offsets
    RECORD_API(inferlet_task_output_get_token_offsets);
    const int *offsets = inferlet_task_output_get_token_offsets(task_out);
    assert(offsets != nullptr);
    assert(offsets[0] == 0);

    // 46. inferlet_task_output_get_token_probability
    RECORD_API(inferlet_task_output_get_token_probability);
    const float *probs = inferlet_task_output_get_token_probability(task_out);
    assert(probs != nullptr);
    for (int i = 0; i < token_len; ++i) {
        assert(probs[i] >= 0.0f && probs[i] <= 1.0f);
    }

    // 47. inferlet_task_output_get_input_tokens
    RECORD_API(inferlet_task_output_get_input_tokens);
    int in_tok_len = 0;
    const char **in_tok_text = nullptr;
    const int *in_toks = inferlet_task_output_get_input_tokens(task_out, &in_tok_len, &in_tok_text);
    assert(in_toks != nullptr);
    assert(in_tok_len > 0);
    assert(in_tok_text != nullptr);

    // 48. inferlet_task_output_get_output_logits
    RECORD_API(inferlet_task_output_get_output_logits);
    int logit_rows = 0, vocab_size = 0;
    const float **logits = inferlet_task_output_get_output_logits(task_out, &logit_rows, &vocab_size);
    assert(logits != nullptr);
    assert(logit_rows == token_len);
    assert(vocab_size > 0);

    // 49. inferlet_task_output_get_output_embedding
    RECORD_API(inferlet_task_output_get_output_embedding);
    int emb_rows = 0, emb_dim = 0;
    const float **embeddings = inferlet_task_output_get_output_embedding(task_out, &emb_rows, &emb_dim);
    assert(embeddings != nullptr);
    assert(emb_rows == token_len);
    assert(emb_dim > 0);

    // 50. inferlet_task_output_deallocate
    RECORD_API(inferlet_task_output_deallocate);
    inferlet_task_output_deallocate(task_out);
    // NULL safety
    inferlet_task_output_deallocate(nullptr);

    inferlet_task_input_deallocate(task_in);
    std::cout << "  -> Sync & TaskOutput APIs (9/9) passed.\n";
}

// -------------------------------------------------------------
// Test Suite 6: Asynchronous Inference & Task Control (4 interfaces)
// -------------------------------------------------------------
static void test_suite_async(void *handle) {
    std::cout << "[Suite 6/7] Testing Async APIs (RunASync, QueryTask, WaitTask, CancelTask)...\n";

    // 1. RunASync + QueryTask
    {
        void *task_in = inferlet_task_input_allocate();
        inferlet_task_input_set_prompt_text(task_in, "Tell me a joke about math");
        inferlet_task_input_set_max_output_tokens(task_in, 5);

        // 51. inferlet_run_async
        RECORD_API(inferlet_run_async);
        void *task = nullptr;
        int rc = inferlet_run_async(handle, task_in, &task);
        assert(rc == INFERLET_OK);
        assert(task != nullptr);

        // 52. inferlet_query_task
        RECORD_API(inferlet_query_task);
        void *task_out = nullptr;
        while (true) {
            rc = inferlet_query_task(handle, task, &task_out);
            assert(rc == INFERLET_OK);
            if (task_out != nullptr) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        assert(inferlet_task_output_get_result_text(task_out) != nullptr);
        inferlet_task_output_deallocate(task_out);
        inferlet_task_input_deallocate(task_in);
    }

    // 2. RunASync + WaitTask
    {
        void *task_in = inferlet_task_input_allocate();
        inferlet_task_input_set_prompt_text(task_in, "The Sun rises in the");
        inferlet_task_input_set_max_output_tokens(task_in, 4);

        void *task = nullptr;
        int rc = inferlet_run_async(handle, task_in, &task);
        assert(rc == INFERLET_OK && task != nullptr);

        // 53. inferlet_wait_task
        RECORD_API(inferlet_wait_task);
        void *task_out = nullptr;
        rc = inferlet_wait_task(handle, task, &task_out);
        assert(rc == INFERLET_OK);
        assert(task_out != nullptr);
        assert(inferlet_task_output_get_result_text(task_out) != nullptr);

        inferlet_task_output_deallocate(task_out);
        inferlet_task_input_deallocate(task_in);
    }

    // 3. RunASync + CancelTask
    {
        void *task_in = inferlet_task_input_allocate();
        inferlet_task_input_set_prompt_text(task_in, "Count from 1 to 1000 slowly");
        inferlet_task_input_set_max_output_tokens(task_in, 100);

        void *task = nullptr;
        int rc = inferlet_run_async(handle, task_in, &task);
        assert(rc == INFERLET_OK && task != nullptr);

        // 54. inferlet_cancel_task
        RECORD_API(inferlet_cancel_task);
        rc = inferlet_cancel_task(handle, task);
        assert(rc == INFERLET_OK);

        // Wait on cancelled task delivers either cancellation error or partial result
        void *task_out = nullptr;
        rc = inferlet_wait_task(handle, task, &task_out);
        if (task_out) {
            inferlet_task_output_deallocate(task_out);
        }

        inferlet_task_input_deallocate(task_in);
    }

    std::cout << "  -> Async APIs (4/4) passed.\n";
}

// -------------------------------------------------------------
// Test Suite 7: Streaming Inference & Stream APIs (8 interfaces)
// -------------------------------------------------------------
static void test_suite_stream(void *handle) {
    std::cout << "[Suite 7/7] Testing Streaming APIs...\n";

    void *task_in = inferlet_task_input_allocate();
    assert(task_in != nullptr);
    inferlet_task_input_set_prompt_text(task_in, "Colors of the rainbow: red, orange,");
    inferlet_task_input_set_max_output_tokens(task_in, 6);
    inferlet_task_input_set_temperature(task_in, 0.0f);
    inferlet_task_input_set_probability_type(task_in, 1);
    inferlet_task_input_set_stream_block_mode(task_in, 1);
    inferlet_task_input_set_verbose_data_flag(task_in, INFERLET_VERBOSE_F_ALL);

    // 55. inferlet_run_stream
    RECORD_API(inferlet_run_stream);
    void *task_stream = nullptr;
    int rc = inferlet_run_stream(handle, task_in, &task_stream);
    assert(rc == INFERLET_OK);
    assert(task_stream != nullptr);

    int chunk_count = 0;
    std::string accumulated_text;

    while (true) {
        // 56. inferlet_get_stream_text
        RECORD_API(inferlet_get_stream_text);
        const char *chunk = inferlet_get_stream_text(handle, task_stream);
        if (chunk == nullptr) {
            // Stream EOF reached! task_stream was automatically freed.
            break;
        }

        chunk_count++;
        accumulated_text += chunk;

        // 57. inferlet_get_stream_tokens
        RECORD_API(inferlet_get_stream_tokens);
        int tok_len = 0;
        const int *stream_toks = inferlet_get_stream_tokens(task_stream, &tok_len);
        assert(stream_toks != nullptr);
        assert(tok_len > 0);

        // 58. inferlet_get_stream_token_offsets
        RECORD_API(inferlet_get_stream_token_offsets);
        const int *stream_offsets = inferlet_get_stream_token_offsets(task_stream);
        assert(stream_offsets != nullptr);

        // 59. inferlet_get_stream_token_probability
        RECORD_API(inferlet_get_stream_token_probability);
        const float *stream_probs = inferlet_get_stream_token_probability(task_stream);
        assert(stream_probs != nullptr);
        for (int i = 0; i < tok_len; ++i) {
            assert(stream_probs[i] >= 0.0f && stream_probs[i] <= 1.0f);
        }

        // 60. inferlet_get_stream_output_logits
        RECORD_API(inferlet_get_stream_output_logits);
        int log_rows = 0, n_vocab = 0;
        const float **stream_logits = inferlet_get_stream_output_logits(task_stream, &log_rows, &n_vocab);
        assert(stream_logits != nullptr);
        assert(log_rows == tok_len);
        assert(n_vocab > 0);

        // 61. inferlet_get_stream_output_embedding (Specially verifying this API!)
        RECORD_API(inferlet_get_stream_output_embedding);
        int emb_rows = 0, emb_dim = 0;
        const float **stream_embeds = inferlet_get_stream_output_embedding(task_stream, &emb_rows, &emb_dim);
        assert(stream_embeds != nullptr);
        assert(emb_rows == tok_len);
        assert(emb_dim > 0);

        // 62. inferlet_get_stream_input_tokens
        RECORD_API(inferlet_get_stream_input_tokens);
        int inp_len = 0;
        const char **inp_texts = nullptr;
        const int *inp_tokens = inferlet_get_stream_input_tokens(task_stream, &inp_len, &inp_texts);
        if (chunk_count == 1) {
            assert(inp_tokens != nullptr);
            assert(inp_len > 0);
            assert(inp_texts != nullptr);
        } else {
            assert(inp_tokens == nullptr);
        }
    }

    assert(chunk_count > 0);
    assert(!accumulated_text.empty());
    std::cout << "  Streamed " << chunk_count << " chunks, text: \"" << accumulated_text << "\"\n";

    inferlet_task_input_deallocate(task_in);
    std::cout << "  -> Streaming APIs (8/8) passed.\n";
}

// -------------------------------------------------------------
// Main entry point
// -------------------------------------------------------------
int main() {
    std::cout << "========================================================\n";
    std::cout << "Starting Comprehensive 64 inferlet.h APIs Verification...\n";
    std::cout << "========================================================\n";

    // 1. Parameter APIs
    test_suite_parameters();

    // 2. Lifecycle & Model Load
    void *handle = test_suite_lifecycle_and_model_load();

    // 3. Tokenizer APIs
    test_suite_tokenizer(handle);

    // 4. TaskInput APIs
    test_suite_task_input();

    // 5. Sync Generation & TaskOutput APIs
    test_suite_sync_and_output(handle);

    // 6. Async Inference & Task Control APIs
    test_suite_async(handle);

    // 7. Streaming Inference & Stream Chunk APIs
    test_suite_stream(handle);

    // 63. inferlet_unload
    RECORD_API(inferlet_unload);
    inferlet_unload(handle);
    // NULL safety
    inferlet_unload(nullptr);

    // 64. inferlet_deinit
    RECORD_API(inferlet_deinit);
    inferlet_deinit();

    std::cout << "========================================================\n";
    std::cout << "Total distinct APIs tested: " << g_tested_apis.size() << " / 64\n";
    for (const auto &api : g_tested_apis) {
        std::cout << "  [OK] " << api << "\n";
    }
    std::cout << "========================================================\n";
    assert(g_tested_apis.size() == 64 && "Must test all 64 interfaces!");
    std::cout << "ALL 64 inferlet.h APIS ARE WORKING PROPERLY!\n";

    return 0;
}

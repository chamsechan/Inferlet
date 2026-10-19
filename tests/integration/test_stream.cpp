#include "inferlet.h"
#include "support/last_error.h"
#include "core/stream_sink.h"
#include <cassert>
#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <chrono>
#include <cmath>

static const char *get_model_path() {
    const char *env = std::getenv("INFERLET_TEST_MODEL");
    if (env && std::strlen(env) > 0) {
        return env;
    }
    return "/home/ubuntu/project/llm-ops-agy/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
}

static void test_stream_basic_block_mode_matches_sync(void *handle) {
    std::cout << "Running test_stream_basic_block_mode_matches_sync...\n";

    const char *prompt = "The quick brown fox jumps over";
    const int max_tokens = 8;

    // 1. Run Synchronous Generation (greedy)
    void *task_in_sync = inferlet_task_input_allocate();
    assert(task_in_sync != nullptr);
    inferlet_task_input_set_prompt_text(task_in_sync, prompt);
    inferlet_task_input_set_max_output_tokens(task_in_sync, max_tokens);
    inferlet_task_input_set_temperature(task_in_sync, 0.0f);
    inferlet_task_input_set_probability_type(task_in_sync, 1);

    void *task_out_sync = nullptr;
    int ret = inferlet_run(handle, task_in_sync, &task_out_sync);
    assert(ret == INFERLET_OK);
    assert(task_out_sync != nullptr);

    const char *sync_text_c = inferlet_task_output_get_result_text(task_out_sync);
    assert(sync_text_c != nullptr);
    std::string sync_text = sync_text_c;

    int sync_tok_len = 0;
    const int *sync_tok_ptr = inferlet_task_output_get_result_tokens(task_out_sync, &sync_tok_len);
    assert(sync_tok_ptr != nullptr);
    assert(sync_tok_len > 0);
    std::vector<int> sync_tokens(sync_tok_ptr, sync_tok_ptr + sync_tok_len);

    const float *sync_prob_ptr = inferlet_task_output_get_token_probability(task_out_sync);
    assert(sync_prob_ptr != nullptr);
    std::vector<float> sync_probs(sync_prob_ptr, sync_prob_ptr + sync_tok_len);

    // 2. Run Stream Generation (greedy, block_mode = 1)
    void *task_in_stream = inferlet_task_input_allocate();
    assert(task_in_stream != nullptr);
    inferlet_task_input_set_prompt_text(task_in_stream, prompt);
    inferlet_task_input_set_max_output_tokens(task_in_stream, max_tokens);
    inferlet_task_input_set_temperature(task_in_stream, 0.0f);
    inferlet_task_input_set_probability_type(task_in_stream, 1);
    inferlet_task_input_set_stream_block_mode(task_in_stream, 1);

    void *task_stream = nullptr;
    ret = inferlet_run_stream(handle, task_in_stream, &task_stream);
    assert(ret == INFERLET_OK);
    assert(task_stream != nullptr);

    std::string stream_text;
    std::vector<int> stream_tokens;
    std::vector<float> stream_probs;
    int chunk_count = 0;

    while (true) {
        const char *chunk_text = inferlet_get_stream_text(handle, task_stream);
        if (chunk_text == nullptr) {
            // Stream EOF reached! task_stream has been automatically freed.
            break;
        }

        chunk_count++;
        assert(std::strlen(chunk_text) > 0);
        stream_text += chunk_text;

        int chunk_tok_len = 0;
        const int *chunk_toks = inferlet_get_stream_tokens(task_stream, &chunk_tok_len);
        assert(chunk_toks != nullptr);
        assert(chunk_tok_len > 0);
        for (int i = 0; i < chunk_tok_len; ++i) {
            stream_tokens.push_back(chunk_toks[i]);
        }

        const int *offsets = inferlet_get_stream_token_offsets(task_stream);
        assert(offsets != nullptr);
        assert(offsets[0] == 0); // First token in chunk starts at offset 0 relative to chunk text

        const float *probs = inferlet_get_stream_token_probability(task_stream);
        assert(probs != nullptr);
        for (int i = 0; i < chunk_tok_len; ++i) {
            stream_probs.push_back(probs[i]);
        }
    }

    std::cout << "Streamed chunks: " << chunk_count << ", text: \"" << stream_text << "\"\n";
    std::cout << "Sync text: \"" << sync_text << "\"\n";

    assert(chunk_count > 0);
    assert(stream_text == sync_text);
    assert(stream_tokens == sync_tokens);
    assert(stream_probs.size() == sync_probs.size());
    for (size_t i = 0; i < stream_probs.size(); ++i) {
        assert(std::abs(stream_probs[i] - sync_probs[i]) < 1e-4f);
    }

    // 3. Post-EOF safety: ticket was freed automatically, subsequent access returns invalid handle
    const char *post_eof = inferlet_get_stream_text(handle, task_stream);
    assert(post_eof == nullptr);
    assert(inferlet::get_last_error_code() == INFERLET_E_INVALID_HANDLE);

    inferlet_task_input_deallocate(task_in_sync);
    inferlet_task_output_deallocate(task_out_sync);
    inferlet_task_input_deallocate(task_in_stream);

    std::cout << "[PASS] test_stream_basic_block_mode_matches_sync\n";
}

static void test_stream_non_blocking_mode(void *handle) {
    std::cout << "Running test_stream_non_blocking_mode (STREAM-001)...\n";

    void *task_in = inferlet_task_input_allocate();
    assert(task_in != nullptr);
    inferlet_task_input_set_prompt_text(task_in, "Artificial intelligence is");
    inferlet_task_input_set_max_output_tokens(task_in, 10);
    inferlet_task_input_set_stream_block_mode(task_in, 0); // Non-blocking

    void *task_stream = nullptr;
    int ret = inferlet_run_stream(handle, task_in, &task_stream);
    assert(ret == INFERLET_OK);
    assert(task_stream != nullptr);

    std::string full_text;
    int empty_chunk_count = 0;
    int valid_chunk_count = 0;

    while (true) {
        const char *text = inferlet_get_stream_text(handle, task_stream);
        if (text == nullptr) {
            // Task completed, EOF reached!
            break;
        }

        if (std::strlen(text) == 0) {
            // Non-blocking: empty string "" indicates task is still running, chunk not ready
            empty_chunk_count++;
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        } else {
            valid_chunk_count++;
            full_text += text;
        }
    }

    std::cout << "Non-blocking empty count: " << empty_chunk_count
              << ", valid chunks: " << valid_chunk_count
              << ", full text: \"" << full_text << "\"\n";

    assert(valid_chunk_count > 0);
    assert(!full_text.empty());

    inferlet_task_input_deallocate(task_in);
    std::cout << "[PASS] test_stream_non_blocking_mode\n";
}

static void test_stream_verbose_and_one_time_input_tokens(void *handle) {
    std::cout << "Running test_stream_verbose_and_one_time_input_tokens...\n";

    void *task_in = inferlet_task_input_allocate();
    assert(task_in != nullptr);
    inferlet_task_input_set_prompt_text(task_in, "Hello world");
    inferlet_task_input_set_max_output_tokens(task_in, 6);
    inferlet_task_input_set_stream_block_mode(task_in, 1);
    inferlet_task_input_set_verbose_data_flag(task_in, INFERLET_VERBOSE_F_INPUT_TOKENS | INFERLET_VERBOSE_F_OUTPUT_LOGITS);

    void *task_stream = nullptr;
    int ret = inferlet_run_stream(handle, task_in, &task_stream);
    assert(ret == INFERLET_OK);
    assert(task_stream != nullptr);

    int chunk_idx = 0;
    while (true) {
        const char *text = inferlet_get_stream_text(handle, task_stream);
        if (text == nullptr) break;

        chunk_idx++;
        int in_len = 0;
        const char **in_texts = nullptr;
        const int *in_toks = inferlet_get_stream_input_tokens(task_stream, &in_len, &in_texts);

        if (chunk_idx == 1) {
            // First output chunk MUST return input tokens!
            assert(in_toks != nullptr);
            assert(in_len > 0);
            assert(in_texts != nullptr);
            assert(in_texts[0] != nullptr);
            std::cout << "First chunk input tokens count: " << in_len
                      << ", first token text: " << in_texts[0] << "\n";
        } else {
            // Subsequent chunks MUST return NULL (one-time API)!
            assert(in_toks == nullptr);
            assert(in_len == 0);
            assert(in_texts == nullptr);
        }

        // Test logits getter
        int logits_len = 0;
        int n_vocab = 0;
        const float **logits = inferlet_get_stream_output_logits(task_stream, &logits_len, &n_vocab);
        assert(logits != nullptr);
        assert(logits_len > 0);
        assert(n_vocab > 0);
        assert(logits[0] != nullptr);
    }

    assert(chunk_idx >= 1);
    inferlet_task_input_deallocate(task_in);
    std::cout << "[PASS] test_stream_verbose_and_one_time_input_tokens\n";
}

static void test_stream_slow_consumer_backpressure(void *handle) {
    std::cout << "Running test_stream_slow_consumer_backpressure...\n";

    void *task_in = inferlet_task_input_allocate();
    assert(task_in != nullptr);
    inferlet_task_input_set_prompt_text(task_in, "Count from one to twenty: 1, 2, 3,");
    inferlet_task_input_set_max_output_tokens(task_in, 20);
    inferlet_task_input_set_stream_block_mode(task_in, 1);

    void *task_stream = nullptr;
    int ret = inferlet_run_stream(handle, task_in, &task_stream);
    assert(ret == INFERLET_OK);
    assert(task_stream != nullptr);

    int chunks = 0;
    std::string text;
    while (true) {
        const char *piece = inferlet_get_stream_text(handle, task_stream);
        if (piece == nullptr) break;

        chunks++;
        text += piece;

        // Slow consumer: sleep 25ms between reads to exercise bounded chunk queue
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }

    assert(chunks > 0);
    assert(!text.empty());
    std::cout << "Slow consumer received " << chunks << " chunks, text: \"" << text << "\"\n";

    inferlet_task_input_deallocate(task_in);
    std::cout << "[PASS] test_stream_slow_consumer_backpressure\n";
}

static void test_stream_cancellation(void *handle) {
    std::cout << "Running test_stream_cancellation...\n";

    void *task_in = inferlet_task_input_allocate();
    assert(task_in != nullptr);
    inferlet_task_input_set_prompt_text(task_in, "Write a long story about a dragon in a magical forest");
    inferlet_task_input_set_max_output_tokens(task_in, 100);
    inferlet_task_input_set_stream_block_mode(task_in, 1);

    void *task_stream = nullptr;
    int ret = inferlet_run_stream(handle, task_in, &task_stream);
    assert(ret == INFERLET_OK);
    assert(task_stream != nullptr);

    // Read 1-2 chunks
    const char *p1 = inferlet_get_stream_text(handle, task_stream);
    assert(p1 != nullptr);

    // Cancel task immediately
    ret = inferlet_cancel_task(handle, task_stream);
    assert(ret == INFERLET_OK);

    // Ticket has been destroyed by CancelTask!
    const char *p2 = inferlet_get_stream_text(handle, task_stream);
    assert(p2 == nullptr);
    assert(inferlet::get_last_error_code() == INFERLET_E_INVALID_HANDLE);

    inferlet_task_input_deallocate(task_in);
    std::cout << "[PASS] test_stream_cancellation\n";
}

static void test_stream_invalid_args_and_validation(void *handle) {
    std::cout << "Running test_stream_invalid_args_and_validation...\n";

    void *task_in = inferlet_task_input_allocate();
    assert(task_in != nullptr);
    inferlet_task_input_set_prompt_text(task_in, "Hello");

    void *stream = nullptr;
    // NULL handle
    int ret = inferlet_run_stream(nullptr, task_in, &stream);
    assert(ret == INFERLET_E_INVALID_HANDLE);

    // NULL task_in
    ret = inferlet_run_stream(handle, nullptr, &stream);
    assert(ret == INFERLET_E_INVALID_ARGUMENT);

    // NULL task_stream output ptr
    ret = inferlet_run_stream(handle, task_in, nullptr);
    assert(ret == INFERLET_E_INVALID_ARGUMENT);

    // Invalid object kinds
    ret = inferlet_run_stream(task_in, task_in, &stream);
    assert(ret == INFERLET_E_INVALID_HANDLE);

    // GetStreamText with NULLs
    const char *text = inferlet_get_stream_text(nullptr, nullptr);
    assert(text == nullptr);

    int len = 0;
    const int *toks = inferlet_get_stream_tokens(nullptr, &len);
    assert(toks == nullptr);

    inferlet_task_input_deallocate(task_in);
    std::cout << "[PASS] test_stream_invalid_args_and_validation\n";
}

int main() {
    std::cout << "=== Inferlet Stream Inference Integration Tests ===\n";

    inferlet_init();

    void *param = inferlet_parameter_allocate();
    assert(param != nullptr);

    const char *model_path = get_model_path();
    std::cout << "Loading model from: " << model_path << "\n";

    void *handle = inferlet_load_from_file(model_path, param);
    assert(handle != nullptr);

    test_stream_basic_block_mode_matches_sync(handle);
    test_stream_non_blocking_mode(handle);
    test_stream_verbose_and_one_time_input_tokens(handle);
    test_stream_slow_consumer_backpressure(handle);
    test_stream_cancellation(handle);
    test_stream_invalid_args_and_validation(handle);

    // Explicit UTF-8 piece aggregation across token boundaries test
    {
        std::cout << "Running test_stream_utf8_aggregation...\n";
        auto core = std::make_shared<inferlet::TaskCore>();
        core->max_queued_chunks = 16;
        inferlet::StreamingSink sink(core);

        // Token 1: partial UTF-8 prefix (first byte of 3-byte Chinese char '你': 0xE4)
        inferlet::TokenStep s1;
        s1.token = 101;
        s1.piece = "\xE4";
        inferlet::Status st = sink.on_token(std::move(s1));
        assert(st.is_ok());
        // Buffer is incomplete UTF-8 -> no chunk emitted yet!
        assert(core->chunks.empty());

        // Token 2: continuation bytes of '你' (0xBD, 0xA0)
        inferlet::TokenStep s2;
        s2.token = 102;
        s2.piece = "\xBD\xA0";
        st = sink.on_token(std::move(s2));
        assert(st.is_ok());
        // Now complete UTF-8 '你' formed! Exactly 1 chunk emitted!
        assert(core->chunks.size() == 1);
        assert(core->chunks.front().result.text == "你");
        assert(core->chunks.front().result.tokens.size() == 2);
        assert(core->chunks.front().result.tokens[0] == 101);
        assert(core->chunks.front().result.tokens[1] == 102);
        assert(core->chunks.front().result.byte_offsets[0] == 0);
        assert(core->chunks.front().result.byte_offsets[1] == 1);

        // Token 3: ASCII '!'
        inferlet::TokenStep s3;
        s3.token = 103;
        s3.piece = "!";
        st = sink.on_token(std::move(s3));
        assert(st.is_ok());
        assert(core->chunks.size() == 2);
        assert(core->chunks.back().result.text == "!");
        assert(core->chunks.back().result.tokens.size() == 1);
        assert(core->chunks.back().result.tokens[0] == 103);
        assert(core->chunks.back().result.byte_offsets[0] == 0);

        sink.on_finish(inferlet::Status::ok());
        std::cout << "[PASS] test_stream_utf8_aggregation\n";
    }

    inferlet_unload(handle);
    inferlet_parameter_deallocate(param);
    inferlet_deinit();

    std::cout << "=== All Stream Inference Integration Tests Passed! ===\n";
    return 0;
}

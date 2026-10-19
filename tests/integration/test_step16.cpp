#include "inferlet.h"
#include "engine/beam_search.h"
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <cassert>
#include <cstdlib>
#include <cmath>
#include <thread>
#include <chrono>

using namespace inferlet;

const std::string MODEL_PATH = [] {
    const char *path = std::getenv("INFERLET_TEST_MODEL");
    return path && *path ? path : "/home/ubuntu/project/llm-ops-agy/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
}();

static void test_toy_vocab_deterministic_math() {
    std::cout << "[Test 1] Toy vocab deterministic hand-calculation math...\n";

    // 5 tokens: 0=A, 1=B, 2=C, 3=D, 4=EOG
    const int vocab_size = 5;
    const int beam_size = 2;
    auto is_eog = [](int tok) { return tok == 4; };

    // Step 1: Prompt logits
    float step1_logits[5] = {4.0f, 2.0f, 0.0f, -1.0f, -5.0f};
    std::vector<double> step1_log_probs;
    BeamSearchExecutor::compute_log_softmax(step1_logits, vocab_size, step1_log_probs);

    // Verify softmax sum is 1.0 (exp sum = 1.0)
    double sum_prob = 0.0;
    for (double lp : step1_log_probs) {
        sum_prob += std::exp(lp);
    }
    assert(std::abs(sum_prob - 1.0) < 1e-5);
    // Token 0 and Token 1 must have highest log probs
    assert(step1_log_probs[0] > step1_log_probs[1]);
    assert(step1_log_probs[1] > step1_log_probs[2]);

    // Initial beams
    std::vector<BeamState> current_beams(2);
    current_beams[0].seq_id = 0;
    current_beams[0].tokens = {0};
    current_beams[0].cumulative_log_prob = step1_log_probs[0];
    current_beams[0].finished = false;

    current_beams[1].seq_id = 1;
    current_beams[1].tokens = {1};
    current_beams[1].cumulative_log_prob = step1_log_probs[1];
    current_beams[1].finished = false;

    // Step 2: Log probs for active beams
    // Beam 0 (token 0) extensions: top token 1 (-0.2) then token 0 (-1.0)
    std::vector<std::vector<double>> step2_log_probs(2);
    step2_log_probs[0] = {-1.0, -0.2, -3.0, -4.0, -2.5};
    // Beam 1 (token 1) extensions: top token 4 (EOG, -0.1)
    step2_log_probs[1] = {-2.0, -3.0, -4.0, -5.0, -0.1};

    std::vector<BeamCandidate> top_candidates;
    BeamSearchExecutor::step_candidates(current_beams, step2_log_probs, vocab_size, beam_size, is_eog, top_candidates);

    assert(top_candidates.size() == 2);
    double cand_0_1 = step1_log_probs[0] - 0.2;
    double cand_0_0 = step1_log_probs[0] - 1.0;
    double cand_1_4 = step1_log_probs[1] - 0.1;

    // Verify hand calculation:
    assert(top_candidates[0].parent_beam_idx == 0 && top_candidates[0].token == 1);
    assert(std::abs(top_candidates[0].new_cumulative_score - cand_0_1) < 1e-6);

    if (cand_0_0 > cand_1_4) {
        assert(top_candidates[1].parent_beam_idx == 0 && top_candidates[1].token == 0);
        assert(std::abs(top_candidates[1].new_cumulative_score - cand_0_0) < 1e-6);
    } else {
        assert(top_candidates[1].parent_beam_idx == 1 && top_candidates[1].token == 4);
        assert(std::abs(top_candidates[1].new_cumulative_score - cand_1_4) < 1e-6);
    }

    // Test length penalty selection
    std::vector<BeamState> final_beams(2);
    // Beam A: 2 tokens, score = -2.0 -> length penalty 1.0: score/2 = -1.0
    final_beams[0].tokens = {0, 1};
    final_beams[0].cumulative_log_prob = -2.0;
    // Beam B: 4 tokens, score = -3.2 -> length penalty 1.0: score/4 = -0.8
    final_beams[1].tokens = {1, 2, 3, 0};
    final_beams[1].cumulative_log_prob = -3.2;

    int best_penalty_1 = BeamSearchExecutor::select_best_beam(final_beams, 1.0f);
    // Beam B has higher average score (-0.8 > -1.0)
    assert(best_penalty_1 == 1);

    // With length penalty 0.0 (raw cumulative score):
    // Beam A: -2.0 > Beam B: -3.2
    int best_penalty_0 = BeamSearchExecutor::select_best_beam(final_beams, 0.0f);
    assert(best_penalty_0 == 0);

    std::cout << "[Test 1] Toy vocab deterministic hand-calculation math PASSED\n";
}

static void test_sampling_parameter_validation() {
    std::cout << "[Test 2] Testing random sampling parameter validation with beam>1...\n";

    inferlet_init();

    void *model = inferlet_load_from_file(MODEL_PATH.c_str(), nullptr);
    assert(model != nullptr);

    // 1. beam_size > 1 combined with temperature > 0 -> should fail on run
    void *input1 = inferlet_task_input_allocate();
    assert(input1 != nullptr);
    assert(inferlet_task_input_set_prompt_text(input1, "Hello") == INFERLET_OK);
    assert(inferlet_task_input_set_beam_size(input1, 2) == INFERLET_OK);
    assert(inferlet_task_input_set_temperature(input1, 0.7f) == INFERLET_OK);
    void *output1 = nullptr;
    assert(inferlet_run(model, input1, &output1) == INFERLET_E_INVALID_ARGUMENT);
    assert(output1 == nullptr);
    inferlet_task_input_deallocate(input1);

    // 2. beam_size > 1 combined with top_p < 1.0 -> should fail on run
    void *input2 = inferlet_task_input_allocate();
    assert(input2 != nullptr);
    assert(inferlet_task_input_set_prompt_text(input2, "Hello") == INFERLET_OK);
    assert(inferlet_task_input_set_beam_size(input2, 2) == INFERLET_OK);
    assert(inferlet_task_input_set_top_p(input2, 0.85f) == INFERLET_OK);
    void *output2 = nullptr;
    assert(inferlet_run(model, input2, &output2) == INFERLET_E_INVALID_ARGUMENT);
    assert(output2 == nullptr);
    inferlet_task_input_deallocate(input2);

    // 3. beam_size > 1 combined with top_k > 0 -> should fail on run
    void *input3 = inferlet_task_input_allocate();
    assert(input3 != nullptr);
    assert(inferlet_task_input_set_prompt_text(input3, "Hello") == INFERLET_OK);
    assert(inferlet_task_input_set_beam_size(input3, 2) == INFERLET_OK);
    assert(inferlet_task_input_set_top_k(input3, 10) == INFERLET_OK);
    void *output3 = nullptr;
    assert(inferlet_run(model, input3, &output3) == INFERLET_E_INVALID_ARGUMENT);
    assert(output3 == nullptr);
    inferlet_task_input_deallocate(input3);

    // 4. beam_size > 1 combined with temperature == 0.0f -> should succeed
    void *input4 = inferlet_task_input_allocate();
    assert(input4 != nullptr);
    assert(inferlet_task_input_set_prompt_text(input4, "Hello") == INFERLET_OK);
    assert(inferlet_task_input_set_beam_size(input4, 2) == INFERLET_OK);
    assert(inferlet_task_input_set_temperature(input4, 0.0f) == INFERLET_OK);
    assert(inferlet_task_input_set_max_output_tokens(input4, 5) == INFERLET_OK);
    void *output4 = nullptr;
    assert(inferlet_run(model, input4, &output4) == INFERLET_OK);
    assert(output4 != nullptr);
    inferlet_task_output_deallocate(output4);
    inferlet_task_input_deallocate(input4);

    // 5. Beam output logits/embeddings are not implemented and must not silently succeed
    void *input5 = inferlet_task_input_allocate();
    assert(input5 != nullptr);
    assert(inferlet_task_input_set_prompt_text(input5, "Hello") == INFERLET_OK);
    assert(inferlet_task_input_set_beam_size(input5, 2) == INFERLET_OK);
    assert(inferlet_task_input_set_temperature(input5, 0.0f) == INFERLET_OK);
    assert(inferlet_task_input_set_verbose_data_flag(
               input5, INFERLET_VERBOSE_F_OUTPUT_LOGITS | INFERLET_VERBOSE_F_OUTPUT_EMBEDDING) == INFERLET_OK);
    void *output5 = nullptr;
    assert(inferlet_run(model, input5, &output5) == INFERLET_E_UNSUPPORTED);
    assert(output5 == nullptr);
    inferlet_task_input_deallocate(input5);

    inferlet_unload(model);
    inferlet_deinit();
    std::cout << "[Test 2] Random sampling parameter validation with beam>1 PASSED\n";
}

static void test_beam1_matches_greedy_identically() {
    std::cout << "[Test 3] Testing beam=1 matches normal greedy generation identically...\n";

    inferlet_init();

    void *model = inferlet_load_from_file(MODEL_PATH.c_str(), nullptr);
    assert(model != nullptr);

    // Run 1: Standard greedy (temperature = 0, default beam_size = 1)
    void *in1 = inferlet_task_input_allocate();
    assert(in1 != nullptr);
    assert(inferlet_task_input_set_prompt_text(in1, "The capital of France is") == INFERLET_OK);
    assert(inferlet_task_input_set_temperature(in1, 0.0f) == INFERLET_OK);
    assert(inferlet_task_input_set_max_output_tokens(in1, 16) == INFERLET_OK);

    void *out1 = nullptr;
    assert(inferlet_run(model, in1, &out1) == INFERLET_OK);
    assert(out1 != nullptr);

    const char *text1 = inferlet_task_output_get_result_text(out1);
    assert(text1 != nullptr);
    std::string result1 = text1;

    // Run 2: Explicit beam_size = 1
    void *in2 = inferlet_task_input_allocate();
    assert(in2 != nullptr);
    assert(inferlet_task_input_set_prompt_text(in2, "The capital of France is") == INFERLET_OK);
    assert(inferlet_task_input_set_beam_size(in2, 1) == INFERLET_OK);
    assert(inferlet_task_input_set_temperature(in2, 0.0f) == INFERLET_OK);
    assert(inferlet_task_input_set_max_output_tokens(in2, 16) == INFERLET_OK);

    void *out2 = nullptr;
    assert(inferlet_run(model, in2, &out2) == INFERLET_OK);
    assert(out2 != nullptr);

    const char *text2 = inferlet_task_output_get_result_text(out2);
    assert(text2 != nullptr);
    std::string result2 = text2;

    std::cout << "Greedy text: [" << result1 << "]\n";
    std::cout << "Beam=1 text: [" << result2 << "]\n";
    assert(result1 == result2);

    inferlet_task_output_deallocate(out1);
    inferlet_task_input_deallocate(in1);
    inferlet_task_output_deallocate(out2);
    inferlet_task_input_deallocate(in2);

    inferlet_unload(model);
    inferlet_deinit();
    std::cout << "[Test 3] beam=1 matches normal greedy generation identically PASSED\n";
}

static void test_beam_search_deterministic_and_clean_kv() {
    std::cout << "[Test 4] Testing beam_size=2 deterministic generation...\n";

    inferlet_init();

    void *model = inferlet_load_from_file(MODEL_PATH.c_str(), nullptr);
    assert(model != nullptr);

    std::string res_run1;
    std::string res_run2;

    for (int run = 0; run < 2; ++run) {
        void *input = inferlet_task_input_allocate();
        assert(input != nullptr);
        assert(inferlet_task_input_set_prompt_text(input, "The largest ocean on Earth is") == INFERLET_OK);
        assert(inferlet_task_input_set_beam_size(input, 2) == INFERLET_OK);
        assert(inferlet_task_input_set_max_output_tokens(input, 16) == INFERLET_OK);

        void *output = nullptr;
        assert(inferlet_run(model, input, &output) == INFERLET_OK);
        assert(output != nullptr);

        const char *text = inferlet_task_output_get_result_text(output);
        assert(text != nullptr);

        if (run == 0) {
            res_run1 = text;
        } else {
            res_run2 = text;
        }

        inferlet_task_output_deallocate(output);
        inferlet_task_input_deallocate(input);
    }

    std::cout << "Beam run 1: [" << res_run1 << "]\n";
    std::cout << "Beam run 2: [" << res_run2 << "]\n";
    assert(!res_run1.empty());
    assert(res_run1 == res_run2);
    assert(res_run1.find("Pacific") != std::string::npos);

    inferlet_unload(model);
    inferlet_deinit();
    std::cout << "[Test 4] beam_size=2 deterministic generation PASSED\n";
}

static void test_beam_search_with_guided_choices() {
    std::cout << "[Test 5] Testing beam_size=2 with guided choices...\n";

    inferlet_init();

    void *model = inferlet_load_from_file(MODEL_PATH.c_str(), nullptr);
    assert(model != nullptr);

    void *input = inferlet_task_input_allocate();
    assert(input != nullptr);
    assert(inferlet_task_input_set_prompt_text(input, "Is water wet? Answer with Yes or No: ") == INFERLET_OK);
    assert(inferlet_task_input_set_beam_size(input, 2) == INFERLET_OK);
    assert(inferlet_task_input_set_max_output_tokens(input, 10) == INFERLET_OK);

    const char *choices[] = {"Yes", "No"};
    assert(inferlet_task_input_set_guided_choices(input, choices, 2) == INFERLET_OK);

    void *output = nullptr;
    assert(inferlet_run(model, input, &output) == INFERLET_OK);
    assert(output != nullptr);

    const char *text = inferlet_task_output_get_result_text(output);
    assert(text != nullptr);
    std::string res = text;
    std::cout << "Guided beam output: [" << res << "]\n";
    assert(res == "Yes" || res == "No");

    inferlet_task_output_deallocate(output);
    inferlet_task_input_deallocate(input);

    inferlet_unload(model);
    inferlet_deinit();
    std::cout << "[Test 5] beam_size=2 with guided choices PASSED\n";
}

static void test_beam_search_cancellation() {
    std::cout << "[Test 6] Testing cancellation during beam search...\n";

    inferlet_init();

    void *model = inferlet_load_from_file(MODEL_PATH.c_str(), nullptr);
    assert(model != nullptr);

    void *input = inferlet_task_input_allocate();
    assert(input != nullptr);
    assert(inferlet_task_input_set_prompt_text(input, "Count from 1 to 1000 in detail: 1, 2, 3, 4,") == INFERLET_OK);
    assert(inferlet_task_input_set_beam_size(input, 2) == INFERLET_OK);
    assert(inferlet_task_input_set_max_output_tokens(input, 200) == INFERLET_OK);

    void *task = nullptr;
    assert(inferlet_run_async(model, input, &task) == INFERLET_OK);
    assert(task != nullptr);

    // Cancel task
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    assert(inferlet_cancel_task(model, task) == INFERLET_OK);

    void *output = nullptr;
    int rc = inferlet_wait_task(model, task, &output);
    std::cout << "WaitTask return code after cancel: " << rc << "\n";
    assert(rc == INFERLET_E_CANCELLED);

    if (output) {
        inferlet_task_output_deallocate(output);
    }
    inferlet_task_input_deallocate(input);

    inferlet_unload(model);
    inferlet_deinit();
    std::cout << "[Test 6] Cancellation during beam search PASSED\n";
}

int main() {
    std::cout << "========== Running Step 16 Beam Search Tests ==========\n";
    test_toy_vocab_deterministic_math();
    test_sampling_parameter_validation();
    test_beam1_matches_greedy_identically();
    test_beam_search_deterministic_and_clean_kv();
    test_beam_search_with_guided_choices();
    test_beam_search_cancellation();
    std::cout << "========== All Step 16 Beam Search Tests PASSED! ==========\n";
    return 0;
}

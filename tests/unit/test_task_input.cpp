#include "inferlet.h"
#include "core/task_input.h"
#include "support/last_error.h"
#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>
#include <limits>

using namespace inferlet;

void test_task_input_defaults() {
    void *handle = inferlet_task_input_allocate();
    assert(handle != nullptr);

    auto *in = checked_cast<TaskInputInternal>(handle, ObjectKind::TaskInput);
    assert(in != nullptr);

    assert(in->sampling.beam_size == 1);
    assert(in->sampling.top_k == 40);
    assert(in->sampling.top_p == 0.95f);
    assert(in->sampling.temperature == 0.8f);
    assert(in->sampling.repetition_penalty == 1.0f);
    assert(in->sampling.max_output_tokens == 256);
    assert(in->sampling.probability_type == 0);
    assert(in->output.verbose_flags == 0);
    assert(!in->output.stream_block);
    assert(in->output.thinking);
    assert(!in->output.keep_special_tokens);
    assert(in->lora_id == 0);
    assert(std::holds_alternative<std::monostate>(in->input));

    inferlet_task_input_deallocate(handle);
    std::cout << "[PASS] test_task_input_defaults\n";
}

void test_task_input_param_validation() {
    void *handle = inferlet_task_input_allocate();
    assert(handle != nullptr);
    auto *in = checked_cast<TaskInputInternal>(handle, ObjectKind::TaskInput);
    assert(in != nullptr);

    // Beam size
    assert(inferlet_task_input_set_beam_size(handle, 0) == INFERLET_E_INVALID_ARGUMENT);
    assert(inferlet_task_input_set_beam_size(handle, -1) == INFERLET_E_INVALID_ARGUMENT);
    assert(inferlet_task_input_set_beam_size(handle, 4) == INFERLET_OK);
    assert(in->sampling.beam_size == 4);

    // Top K
    assert(inferlet_task_input_set_top_k(handle, -5) == INFERLET_E_INVALID_ARGUMENT);
    assert(inferlet_task_input_set_top_k(handle, 50) == INFERLET_OK);
    assert(in->sampling.top_k == 50);
    assert(inferlet_task_input_set_top_k(handle, 0) == INFERLET_OK);
    assert(in->sampling.top_k == 40); // 0 resets to default

    // Top P
    assert(inferlet_task_input_set_top_p(handle, -0.1f) == INFERLET_E_INVALID_ARGUMENT);
    assert(inferlet_task_input_set_top_p(handle, 0.0f) == INFERLET_E_INVALID_ARGUMENT);
    assert(inferlet_task_input_set_top_p(handle, 1.1f) == INFERLET_E_INVALID_ARGUMENT);
    assert(inferlet_task_input_set_top_p(handle, 0.90f) == INFERLET_OK);
    assert(in->sampling.top_p == 0.90f);

    // Temperature
    assert(inferlet_task_input_set_temperature(handle, -0.5f) == INFERLET_E_INVALID_ARGUMENT);
    assert(inferlet_task_input_set_temperature(handle, 0.0f) == INFERLET_OK);
    assert(in->sampling.temperature == 0.0f);
    assert(inferlet_task_input_set_temperature(handle, 0.7f) == INFERLET_OK);
    assert(in->sampling.temperature == 0.7f);

    // Repetition Penalty
    assert(inferlet_task_input_set_repetition_penalty(handle, -0.1f) == INFERLET_E_INVALID_ARGUMENT);
    assert(inferlet_task_input_set_repetition_penalty(handle, 0.0f) == INFERLET_E_INVALID_ARGUMENT);
    assert(inferlet_task_input_set_repetition_penalty(handle, 1.2f) == INFERLET_OK);
    assert(in->sampling.repetition_penalty == 1.2f);

    // Max Output Tokens
    assert(inferlet_task_input_set_max_output_tokens(handle, 0) == INFERLET_E_INVALID_ARGUMENT);
    assert(inferlet_task_input_set_max_output_tokens(handle, -100) == INFERLET_E_INVALID_ARGUMENT);
    assert(inferlet_task_input_set_max_output_tokens(handle, 2048) == INFERLET_OK);
    assert(in->sampling.max_output_tokens == 2048);

    // Probability Type
    assert(inferlet_task_input_set_probability_type(handle, -1) == INFERLET_E_INVALID_ARGUMENT);
    assert(inferlet_task_input_set_probability_type(handle, 3) == INFERLET_E_INVALID_ARGUMENT);
    assert(inferlet_task_input_set_probability_type(handle, 2) == INFERLET_OK);
    assert(in->sampling.probability_type == 2);

    // Verbose flag
    assert(inferlet_task_input_set_verbose_data_flag(handle, 0x07) == INFERLET_OK);
    assert(in->output.verbose_flags == 0x07);
    assert(inferlet_task_input_set_verbose_data_flag(handle, 0x10) == INFERLET_E_INVALID_ARGUMENT);

    // Stream block mode
    assert(inferlet_task_input_set_stream_block_mode(handle, -1) == INFERLET_E_INVALID_ARGUMENT);
    assert(inferlet_task_input_set_stream_block_mode(handle, 2) == INFERLET_E_INVALID_ARGUMENT);
    assert(inferlet_task_input_set_stream_block_mode(handle, 1) == INFERLET_OK);
    assert(in->output.stream_block);

    // Thinking mode
    assert(inferlet_task_input_set_thinking_mode(handle, -1) == INFERLET_E_INVALID_ARGUMENT);
    assert(inferlet_task_input_set_thinking_mode(handle, 0) == INFERLET_OK);
    assert(!in->output.thinking);
    assert(inferlet_task_input_set_thinking_mode(handle, 1) == INFERLET_OK);
    assert(in->output.thinking);

    // Special token mode
    assert(inferlet_task_input_set_special_token_mode(handle, -1) == INFERLET_E_INVALID_ARGUMENT);
    assert(inferlet_task_input_set_special_token_mode(handle, 1) == INFERLET_OK);
    assert(in->output.keep_special_tokens);

    // LoRA id
    assert(inferlet_task_input_set_low_rank_adapter_id(handle, -1) == INFERLET_E_INVALID_ARGUMENT);
    assert(inferlet_task_input_set_low_rank_adapter_id(handle, 2) == INFERLET_OK);
    assert(in->lora_id == 2);

    // Visual encoder
    assert(inferlet_task_input_set_visual_encoder(handle, "clip-vit-base") == INFERLET_OK);
    assert(in->visual_encoder_id == "clip-vit-base");

    inferlet_task_input_deallocate(handle);
    std::cout << "[PASS] test_task_input_param_validation\n";
}

void test_task_input_deep_copy() {
    void *handle = inferlet_task_input_allocate();
    assert(handle != nullptr);
    auto *in = checked_cast<TaskInputInternal>(handle, ObjectKind::TaskInput);

    // 1. Text prompt deep copy
    {
        char *temp_str = strdup("Hello world, this is a deep copy test!");
        assert(inferlet_task_input_set_prompt_text(handle, temp_str) == INFERLET_OK);
        // Overwrite and free temp_str
        memset(temp_str, 0, strlen(temp_str));
        free(temp_str);

        auto *p = std::get_if<PromptTextInput>(&in->input);
        assert(p != nullptr);
        assert(p->text == "Hello world, this is a deep copy test!");
    }

    // 2. Tokens deep copy
    {
        int *tokens = (int *)malloc(5 * sizeof(int));
        for (int i = 0; i < 5; ++i) tokens[i] = 100 + i;
        assert(inferlet_task_input_set_prompt_tokens(handle, tokens, 5) == INFERLET_OK);
        memset(tokens, 0, 5 * sizeof(int));
        free(tokens);

        auto *tok = std::get_if<PromptTokensInput>(&in->input);
        assert(tok != nullptr);
        assert(tok->tokens.size() == 5);
        for (int i = 0; i < 5; ++i) {
            assert(tok->tokens[i] == 100 + i);
        }
    }

    // 3. Chat history deep copy
    {
        inferlet_chat_history chat[2];
        chat[0].system = strdup("You are a helpful assistant.");
        chat[0].query = strdup("What is Paris?");
        chat[0].response = strdup("The capital of France.");
        chat[1].system = nullptr;
        chat[1].query = strdup("Tell me more.");
        chat[1].response = nullptr;

        assert(inferlet_task_input_set_chat_history(handle, chat, 2) == INFERLET_OK);

        // Wipe and free source
        free((void *)chat[0].system);
        free((void *)chat[0].query);
        free((void *)chat[0].response);
        free((void *)chat[1].query);

        auto *c = std::get_if<ChatInput>(&in->input);
        assert(c != nullptr);
        assert(c->history.size() == 2);
        assert(c->history[0].system == "You are a helpful assistant.");
        assert(c->history[0].query == "What is Paris?");
        assert(c->history[0].response == "The capital of France.");
        assert(c->history[1].query == "Tell me more.");
        assert(c->history[1].response.empty());
    }

    // 4. MultiModal Chat deep copy
    {
        inferlet_multimodal_chat_history mm_chat[1] = {};
        mm_chat[0].role = strdup("user");

        inferlet_multimodal_chat_content contents[1] = {};
        contents[0].type = "image";
        uint8_t dummy_img[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        contents[0].data = dummy_img;
        contents[0].size = 16;
        contents[0].grid_t = 1;
        contents[0].grid_h = 2;
        contents[0].grid_w = 2;
        contents[0].data_type = 4; // uint8
        contents[0].cache_key = "img_hash_123";
        contents[0].cache_times = 2;
        contents[0].second_per_grid_t = 0.0f;
        contents[0].fps = 24.0f;
        contents[0].ext = nullptr;

        mm_chat[0].content = contents;
        mm_chat[0].length = 1;

        assert(inferlet_task_input_set_multimodal_chat_history(handle, mm_chat, 1) == INFERLET_OK);

        free((void *)mm_chat[0].role);

        auto *mc = std::get_if<MultiModalChatInput>(&in->input);
        assert(mc != nullptr);
        assert(mc->history.size() == 1);
        assert(mc->history[0].role == "user");
        assert(mc->history[0].contents.size() == 1);
        assert(mc->history[0].contents[0].type == "image");
        assert(mc->history[0].contents[0].binary_data.size() == 16);
        assert(mc->history[0].contents[0].binary_data[0] == 1);
        assert(mc->history[0].contents[0].binary_data[15] == 16);
    }

    // 5. Guided choices and Tools deep copy
    {
        const char *choices[2] = {"positive", "negative"};
        assert(inferlet_task_input_set_guided_choices(handle, choices, 2) == INFERLET_OK);
        assert(in->guided_choices.size() == 2);
        assert(in->guided_choices[0] == "positive");
        assert(in->guided_choices[1] == "negative");

        const char *tools[1] = {"{\"type\": \"function\", \"function\": {\"name\": \"foo\"}}"};
        assert(inferlet_task_input_set_tools(handle, tools, 1) == INFERLET_OK);
        assert(in->tools.size() == 1);
        assert(in->tools[0]["type"] == "function");
    }

    inferlet_task_input_deallocate(handle);
    std::cout << "[PASS] test_task_input_deep_copy\n";
}

void test_task_input_variant_mutual_exclusion() {
    void *handle = inferlet_task_input_allocate();
    assert(handle != nullptr);
    auto *in = checked_cast<TaskInputInternal>(handle, ObjectKind::TaskInput);

    // 1. Set text
    inferlet_task_input_set_prompt_text(handle, "first prompt");
    assert(std::holds_alternative<PromptTextInput>(in->input));

    // 2. Overwrite with tokens -> text is replaced
    int tokens[2] = {1, 2};
    inferlet_task_input_set_prompt_tokens(handle, tokens, 2);
    assert(std::holds_alternative<PromptTokensInput>(in->input));
    assert(!std::holds_alternative<PromptTextInput>(in->input));

    // 3. Overwrite with chat
    inferlet_chat_history chat[1];
    chat[0].system = nullptr;
    chat[0].query = "hi";
    chat[0].response = nullptr;
    inferlet_task_input_set_chat_history(handle, chat, 1);
    assert(std::holds_alternative<ChatInput>(in->input));
    assert(!std::holds_alternative<PromptTokensInput>(in->input));

    // 4. Overwrite with multimodal
    inferlet_multimodal mm[1] = {};
    const char *text = "sample";
    mm[0].prompt_text = text;
    mm[0].feature = nullptr;
    mm[0].size = 0;
    inferlet_task_input_set_multimodal(handle, mm, 1);
    assert(std::holds_alternative<LegacyMultiModalInput>(in->input));
    assert(!std::holds_alternative<ChatInput>(in->input));

    // 5. Overwrite back to text
    inferlet_task_input_set_prompt_text(handle, "final prompt");
    assert(std::holds_alternative<PromptTextInput>(in->input));
    assert(std::get<PromptTextInput>(in->input).text == "final prompt");

    inferlet_task_input_deallocate(handle);
    std::cout << "[PASS] test_task_input_variant_mutual_exclusion\n";
}

void test_task_input_create_request_snapshot() {
    void *handle = inferlet_task_input_allocate();
    assert(handle != nullptr);
    auto *in = checked_cast<TaskInputInternal>(handle, ObjectKind::TaskInput);

    // Cannot create request when no variant set
    TaskRequest req;
    Status st = in->create_request(req);
    assert(!st.is_ok());
    assert(st.code == INFERLET_E_INVALID_ARGUMENT);

    // Set valid text input
    inferlet_task_input_set_prompt_text(handle, "Original prompt");
    inferlet_task_input_set_temperature(handle, 0.42f);
    inferlet_task_input_set_top_k(handle, 20);

    st = in->create_request(req);
    assert(st.is_ok());

    assert(req.sampling.temperature == 0.42f);
    assert(req.sampling.top_k == 20);
    assert(std::holds_alternative<PromptTextInput>(req.input));
    assert(std::get<PromptTextInput>(req.input).text == "Original prompt");

    // Modify TaskInput after snapshot taken
    inferlet_task_input_set_prompt_text(handle, "Modified prompt");
    inferlet_task_input_set_temperature(handle, 0.99f);
    inferlet_task_input_set_top_k(handle, 80);

    // Request snapshot MUST NOT change
    assert(req.sampling.temperature == 0.42f);
    assert(req.sampling.top_k == 20);
    assert(std::get<PromptTextInput>(req.input).text == "Original prompt");

    // Deallocate task input entirely; req should remain valid
    inferlet_task_input_deallocate(handle);
    assert(req.sampling.temperature == 0.42f);
    assert(req.sampling.top_k == 20);
    assert(std::get<PromptTextInput>(req.input).text == "Original prompt");

    std::cout << "[PASS] test_task_input_create_request_snapshot\n";
}

void test_nonfinite_sampling_parameters() {
    void *input = inferlet_task_input_allocate();
    assert(input != nullptr);
    assert(inferlet_task_input_set_top_p(input, 0.5f) == INFERLET_OK);
    assert(inferlet_task_input_set_temperature(input, 0.5f) == INFERLET_OK);
    assert(inferlet_task_input_set_repetition_penalty(input, 1.2f) == INFERLET_OK);
    for (float value : {std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::infinity(),
                        -std::numeric_limits<float>::infinity()}) {
        assert(inferlet_task_input_set_top_p(input, value) == INFERLET_E_INVALID_ARGUMENT);
        assert(inferlet_task_input_set_temperature(input, value) == INFERLET_E_INVALID_ARGUMENT);
        assert(inferlet_task_input_set_repetition_penalty(input, value) == INFERLET_E_INVALID_ARGUMENT);
    }
    auto *in = checked_cast<TaskInputInternal>(input, ObjectKind::TaskInput);
    assert(in->sampling.top_p == 0.5f);
    assert(in->sampling.temperature == 0.5f);
    assert(in->sampling.repetition_penalty == 1.2f);
    inferlet_task_input_deallocate(input);
}

int main() {
    std::cout << "Starting TaskInput unit tests...\n";
    test_task_input_defaults();
    test_task_input_param_validation();
    test_nonfinite_sampling_parameters();
    test_task_input_deep_copy();
    test_task_input_variant_mutual_exclusion();
    test_task_input_create_request_snapshot();
    std::cout << "All TaskInput unit tests passed!\n";
    return 0;
}

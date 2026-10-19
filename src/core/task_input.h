#ifndef INFERLET_CORE_TASK_INPUT_H_INCLUDED
#define INFERLET_CORE_TASK_INPUT_H_INCLUDED

#include "support/object_registry.h"
#include "support/status.h"
#include "inferlet.h"
#include <string>
#include <vector>
#include <variant>
#include <cstdint>
#include <nlohmann/json.hpp>

namespace inferlet {

struct SamplingConfig {
    int beam_size = 1;
    int top_k = 40;
    float top_p = 0.95f;
    float temperature = 0.8f;
    float repetition_penalty = 1.0f;
    int repeat_last_n = 64;
    int max_output_tokens = 256;
    int probability_type = 0;
    uint64_t seed = 0;
    bool custom_temperature = false;
    bool custom_top_p = false;
    bool custom_top_k = false;
};

struct OutputOptions {
    int verbose_flags = 0;
    bool stream_block = false;
    bool thinking = true;
    bool keep_special_tokens = false;
};

struct PromptTextInput {
    std::string text;
};

struct PromptTokensInput {
    std::vector<int> tokens;
};

struct ChatMessage {
    std::string system;
    std::string query;
    std::string response;
};

struct ChatInput {
    std::vector<ChatMessage> history;
};

struct LegacyMultiModalItem {
    std::string prompt_text;
    std::vector<uint8_t> feature;
    int size = 0;
};

struct LegacyMultiModalInput {
    std::vector<LegacyMultiModalItem> items;
};

struct MultiModalChatContentExt {
    std::vector<int> frame_index;
    std::string language;
    int frame_num = 0;
    int packing_num = 1;
    int slice_num = 0;
};

struct MultiModalChatContent {
    std::string type;
    std::string text_data;
    std::vector<uint8_t> binary_data;
    std::string cache_key;
    MultiModalChatContentExt ext;
    int size = 0;
    int grid_t = 0;
    int grid_h = 0;
    int grid_w = 0;
    int data_type = 0;
    float second_per_grid_t = 0.0f;
    int cache_times = 0;
    float fps = 24.0f;
};

struct MultiModalChatMessage {
    std::string role;
    std::vector<MultiModalChatContent> contents;
};

struct MultiModalChatInput {
    std::vector<MultiModalChatMessage> history;
};

using InputVariant = std::variant<
    std::monostate,
    PromptTextInput,
    PromptTokensInput,
    ChatInput,
    LegacyMultiModalInput,
    MultiModalChatInput
>;

struct TaskRequest {
    InputVariant input;
    SamplingConfig sampling;
    OutputOptions output;
    int lora_id = 0;
    std::vector<std::string> guided_choices;
    std::vector<nlohmann::json> tools;
    std::string visual_encoder_id;
};

struct TaskInputInternal {
    ObjectHeader header{INFERLET_OBJECT_MAGIC, ObjectKind::TaskInput, INFERLET_ABI_VERSION};
    InputVariant input;
    SamplingConfig sampling;
    OutputOptions output;
    int lora_id = 0;
    std::vector<std::string> guided_choices;
    std::vector<nlohmann::json> tools;
    std::string visual_encoder_id;

    Status set_prompt_text(const char *text);
    Status set_prompt_tokens(const int *tokens, int length);
    Status set_chat_history(const inferlet_chat_history history[], int length);
    Status set_multimodal(const inferlet_multimodal mm[], int length);
    Status set_multimodal_chat(const inferlet_multimodal_chat_history history[], int length);

    Status set_beam_size(int size);
    Status set_top_k(int k);
    Status set_top_p(float p);
    Status set_temperature(float temp);
    Status set_repetition_penalty(float penalty);
    Status set_max_output_tokens(int max_tokens);
    Status set_probability_type(int type);
    Status set_verbose_flag(int flag);
    Status set_stream_block(int block);
    Status set_thinking(int thinking);
    Status set_special_token(int keep_special);
    Status set_guided_choices(const char *choices[], int num);
    Status set_tools(const char *tools[], int num);
    Status set_visual_encoder(const char *id);

    Status create_request(TaskRequest &out_req) const;
};

} // namespace inferlet

#endif // INFERLET_CORE_TASK_INPUT_H_INCLUDED

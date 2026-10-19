#ifndef INFERLET_ENGINE_PROMPT_BUILDER_H_INCLUDED
#define INFERLET_ENGINE_PROMPT_BUILDER_H_INCLUDED

#include "core/task_input.h"
#include "core/model_registry.h"
#include "support/status.h"
#include <vector>
#include <string>
#include <utility>

namespace inferlet {

struct InputTokenInfo {
    std::vector<int> tokens;
    std::vector<std::string> text_storage;
    std::vector<const char *> text_ptrs;

    void update_ptrs() {
        text_ptrs.resize(text_storage.size());
        for (size_t i = 0; i < text_storage.size(); ++i) {
            text_ptrs[i] = text_storage[i].c_str();
        }
    }
};

struct MtmdPrompt;

struct PreparedPrompt {
    std::vector<int> tokens;
    InputTokenInfo verbose_input;
    std::shared_ptr<MtmdPrompt> multimodal; // Owns preprocessed image chunks
    std::string formatted_text;             // Formatted text if generated from chat template
};

class PromptBuilder {
public:
    static Status build(const TaskRequest &request,
                        const std::shared_ptr<SharedModel> &model,
                        PreparedPrompt &out_prompt);

    // Unfold ChatHistory according to section 6.1 rules into normalized messages
    static Status unfold_chat_history(const std::vector<ChatMessage> &history,
                                      std::vector<std::pair<std::string, std::string>> &out_messages);

    // Chat template application helper
    static Status apply_chat_template(const std::vector<ChatMessage> &history,
                                      const char *custom_template_or_null,
                                      const std::shared_ptr<SharedModel> &model,
                                      std::string &out_formatted_text,
                                      const std::vector<nlohmann::json> *tools = nullptr);
};

} // namespace inferlet

#endif // INFERLET_ENGINE_PROMPT_BUILDER_H_INCLUDED

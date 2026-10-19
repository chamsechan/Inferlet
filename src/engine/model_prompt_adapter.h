#ifndef INFERLET_ENGINE_MODEL_PROMPT_ADAPTER_H_INCLUDED
#define INFERLET_ENGINE_MODEL_PROMPT_ADAPTER_H_INCLUDED

#include "core/task_input.h"
#include "support/status.h"
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

struct llama_model;

namespace inferlet {

struct SharedModel;

struct ModelCapabilities {
    std::string architecture;
    std::string model_name;
    bool supports_thinking = false;
    bool supports_tools = false;
    bool supports_lora = true;
    bool supports_vision = false;
};

class ModelPromptAdapter {
public:
    // Query model metadata and determine capabilities
    static Status detect_capabilities(const llama_model *model, ModelCapabilities &out_caps);

    // Validate whether the TaskRequest features are supported by the model
    static Status validate_request_capabilities(const TaskRequest &request, const SharedModel &model);

    // Format tool JSON objects into a prompt specification block
    static std::string format_tools_description(const std::vector<nlohmann::json> &tools);
};

} // namespace inferlet

#endif // INFERLET_ENGINE_MODEL_PROMPT_ADAPTER_H_INCLUDED

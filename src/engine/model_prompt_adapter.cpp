#include "model_prompt_adapter.h"
#include "core/model_registry.h"
#include "engine/llama_engine.h"
#include "engine/lora_registry.h"
#include "support/logging.h"

namespace inferlet {

Status ModelPromptAdapter::detect_capabilities(const llama_model *model, ModelCapabilities &out_caps) {
    if (!model) {
        return Status::invalid_handle("Model is NULL in detect_capabilities");
    }
    return LlamaEngine::get_model_capabilities(model, out_caps);
}

Status ModelPromptAdapter::validate_request_capabilities(const TaskRequest &request, const SharedModel &model) {
    // Section 6.8: Guided choices and tools cannot be specified together in v1
    if (!request.guided_choices.empty() && !request.tools.empty()) {
        return Status::unsupported("Guided choices and tools cannot be specified together");
    }

    ModelCapabilities caps;
    if (model.model) {
        Status st = detect_capabilities(model.model.get(), caps);
        if (!st) return st;
    }

    const bool is_multimodal = std::holds_alternative<LegacyMultiModalInput>(request.input) ||
                               std::holds_alternative<MultiModalChatInput>(request.input);
    if (is_multimodal && !model.multimodal) {
        return Status::unsupported("Multimodal inference requires a supported vision model and visual projector");
    }
    if (is_multimodal && request.sampling.beam_size > 1) {
        return Status::unsupported("Image input currently requires beam_size=1");
    }

    // Tools validation
    if (!request.tools.empty()) {
        if (model.config.tokenizer_only) {
            return Status::unsupported("Tools are not supported in tokenizer-only mode");
        }
        if (std::holds_alternative<PromptTokensInput>(request.input)) {
            return Status::unsupported("Tools cannot be applied to raw token inputs");
        }
        if (!caps.supports_tools) {
            return Status::unsupported("Model does not support tools or lacks a chat template");
        }
    }

    if (caps.supports_thinking && !request.output.thinking) {
        return Status::unsupported("Disabling thinking mode is not supported by the current chat-template backend");
    }
    if (model.model && !caps.supports_thinking) {
        log_debug("Thinking mode requested (" + std::string(request.output.thinking ? "true" : "false") +
                  "), but model architecture '" + caps.architecture + "' does not support thinking; ignoring per specification");
    }

    // LoRA validation
    if (request.lora_id > 0) {
        if (model.config.tokenizer_only) {
            return Status::unsupported("LoRA is not supported in tokenizer-only mode");
        }
        if (!model.lora_registry || !model.lora_registry->has_adapter(request.lora_id)) {
            return Status::unsupported("LoRA adapter ID " + std::to_string(request.lora_id) + " not registered on model");
        }
    }

    return Status::ok();
}

std::string ModelPromptAdapter::format_tools_description(const std::vector<nlohmann::json> &tools) {
    if (tools.empty()) return "";

    nlohmann::json arr = nlohmann::json::array();
    for (const auto &t : tools) {
        arr.push_back(t);
    }

    std::string out;
    out += "# Tools\n\n";
    out += "You may call one or more functions to assist with the user query.\n";
    out += "You are provided with function signatures within <tools></tools> XML tags:\n";
    out += "<tools>\n";
    out += arr.dump(2);
    out += "\n</tools>\n\n";
    out += "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n";
    out += "<tool_call>\n";
    out += "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n";
    out += "</tool_call>";
    return out;
}

} // namespace inferlet

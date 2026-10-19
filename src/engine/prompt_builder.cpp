#include "prompt_builder.h"
#include "engine/llama_engine.h"
#include "engine/token_codec.h"
#include "engine/model_prompt_adapter.h"
#include "multimodal/mtmd_backend.h"
#include "support/logging.h"
#include <cstring>

namespace inferlet {

Status PromptBuilder::unfold_chat_history(const std::vector<ChatMessage> &history,
                                          std::vector<std::pair<std::string, std::string>> &out_messages) {
    if (history.empty()) {
        return Status::invalid_argument("Chat history cannot be empty");
    }

    std::string system_prompt;
    bool has_system = false;
    for (const auto &item : history) {
        if (!item.system.empty()) {
            if (!has_system) {
                system_prompt = item.system;
                has_system = true;
            } else if (system_prompt != item.system) {
                return Status::invalid_argument("Conflicting system prompt in chat history");
            }
        }
    }

    out_messages.clear();
    if (has_system) {
        out_messages.push_back({"system", system_prompt});
    }

    for (size_t i = 0; i < history.size(); ++i) {
        const auto &item = history[i];
        if (item.query.empty() && item.response.empty()) {
            continue;
        }

        if (!item.query.empty()) {
            out_messages.push_back({"user", item.query});
        }

        if (!item.response.empty()) {
            if (out_messages.empty() || out_messages.back().first != "user") {
                return Status::invalid_argument("Assistant response must follow a user query in chat history");
            }
            out_messages.push_back({"assistant", item.response});
        }
    }

    if (out_messages.empty()) {
        return Status::invalid_argument("Chat history contains no messages");
    }
    if (out_messages.back().first == "assistant") {
        return Status::invalid_argument("Chat history cannot end with complete assistant response");
    }
    if (out_messages.back().first != "user") {
        return Status::invalid_argument("Chat history must end with user query");
    }

    return Status::ok();
}

Status PromptBuilder::apply_chat_template(const std::vector<ChatMessage> &history,
                                          const char *custom_template_or_null,
                                          const std::shared_ptr<SharedModel> &model,
                                          std::string &out_formatted_text,
                                          const std::vector<nlohmann::json> *tools) {
    std::vector<std::pair<std::string, std::string>> chat_msgs;
    Status st = unfold_chat_history(history, chat_msgs);
    if (!st) return st;

    if (tools && !tools->empty()) {
        std::string tools_desc = ModelPromptAdapter::format_tools_description(*tools);
        if (!chat_msgs.empty() && chat_msgs[0].first == "system") {
            chat_msgs[0].second += "\n\n" + tools_desc;
        } else {
            chat_msgs.insert(chat_msgs.begin(), {"system", tools_desc});
        }
    }

    const char *tmpl = custom_template_or_null;
    if (!tmpl || std::strlen(tmpl) == 0) {
        if (!model) {
            return Status::unsupported("No model loaded to retrieve chat template");
        }
        {
            std::unique_lock<std::mutex> lock(model->mutex);
            if (model->load_state == LoadState::Loading) {
                model->load_cv.wait(lock, [&model]() {
                    return model->load_state != LoadState::Loading;
                });
            }
            if (model->load_state == LoadState::Failed) {
                return model->load_status.is_ok() ? Status::model_load() : model->load_status;
            }
        }
        if (!model->model) {
            return Status::unsupported("No model loaded to retrieve chat template");
        }
        tmpl = LlamaEngine::get_model_chat_template(model->model.get());
    }

    if (!tmpl || std::strlen(tmpl) == 0) {
        return Status::unsupported("Model does not contain a built-in chat template");
    }

    int32_t res = LlamaEngine::apply_chat_template(tmpl, chat_msgs, /*add_ass=*/true, out_formatted_text);
    if (res < 0) {
        return Status::internal("Failed to apply chat template to messages");
    }

    return Status::ok();
}

Status PromptBuilder::build(const TaskRequest &request,
                            const std::shared_ptr<SharedModel> &model,
                            PreparedPrompt &out_prompt) {
    if (!model) {
        return Status::invalid_handle("Model is NULL");
    }
    {
        std::unique_lock<std::mutex> lock(model->mutex);
        if (model->load_state == LoadState::Loading) {
            model->load_cv.wait(lock, [&model]() {
                return model->load_state != LoadState::Loading;
            });
        }
        if (model->load_state == LoadState::Failed) {
            return model->load_status.is_ok() ? Status::model_load() : model->load_status;
        }
    }
    if (!model->vocab) {
        return Status::invalid_handle("Model or model vocab is not loaded");
    }

    Status st_cap = ModelPromptAdapter::validate_request_capabilities(request, *model);
    if (!st_cap) return st_cap;

    out_prompt.tokens.clear();
    out_prompt.multimodal.reset();
    out_prompt.formatted_text.clear();
    out_prompt.verbose_input.tokens.clear();
    out_prompt.verbose_input.text_storage.clear();
    out_prompt.verbose_input.text_ptrs.clear();

    if (std::holds_alternative<std::monostate>(request.input)) {
        return Status::invalid_argument("No input variant specified in TaskRequest");
    }

    if (auto *text_in = std::get_if<PromptTextInput>(&request.input)) {
        if (!request.tools.empty()) {
            std::vector<ChatMessage> history = { { "", text_in->text, "" } };
            Status st = apply_chat_template(history, nullptr, model, out_prompt.formatted_text, &request.tools);
            if (!st) return st;
            st = TokenCodec::tokenize(model->vocab, out_prompt.formatted_text, /*add_special=*/false, /*parse_special=*/true, out_prompt.tokens);
            if (!st) return st;
        } else if (text_in->text.empty()) {
            int bos = LlamaEngine::get_vocab_bos(model->vocab);
            if (bos != -1 && !LlamaEngine::is_vocab_eog(model->vocab, bos)) {
                out_prompt.tokens.push_back(bos);
            } else {
                return Status::invalid_argument("Prompt text is empty and model has no valid BOS token");
            }
        } else {
            Status st = TokenCodec::tokenize(model->vocab, text_in->text, /*add_special=*/true, /*parse_special=*/false, out_prompt.tokens);
            if (!st) return st;
        }
    } else if (auto *tokens_in = std::get_if<PromptTokensInput>(&request.input)) {
        if (tokens_in->tokens.empty()) {
            int bos = LlamaEngine::get_vocab_bos(model->vocab);
            if (bos != -1 && !LlamaEngine::is_vocab_eog(model->vocab, bos)) {
                out_prompt.tokens.push_back(bos);
            } else {
                return Status::invalid_argument("Prompt tokens array is empty and model has no valid BOS token");
            }
        } else {
            int32_t n_vocab = LlamaEngine::get_vocab_n_tokens(model->vocab);
            for (int t : tokens_in->tokens) {
                if (t < 0 || (n_vocab > 0 && t >= n_vocab)) {
                    return Status::invalid_argument("Token ID out of vocab range: " + std::to_string(t));
                }
            }
            out_prompt.tokens = tokens_in->tokens;
        }
    } else if (auto *chat_in = std::get_if<ChatInput>(&request.input)) {
        Status st = apply_chat_template(chat_in->history, nullptr, model, out_prompt.formatted_text,
                                        request.tools.empty() ? nullptr : &request.tools);
        if (!st) return st;

        st = TokenCodec::tokenize(model->vocab, out_prompt.formatted_text, /*add_special=*/false, /*parse_special=*/true, out_prompt.tokens);
        if (!st) return st;
    } else if (std::holds_alternative<LegacyMultiModalInput>(request.input) ||
               std::holds_alternative<MultiModalChatInput>(request.input)) {
        if (!model->multimodal) {
            return Status::unsupported("Model does not support multimodal inputs");
        }

        Status st = model->multimodal->prepare(request, *model, out_prompt);
        if (!st) return st;
    } else {
        return Status::invalid_argument("Unknown input variant");
    }

    // Populate verbose input token info if requested
    if (request.output.verbose_flags & INFERLET_VERBOSE_F_INPUT_TOKENS) {
        out_prompt.verbose_input.tokens = out_prompt.tokens;
        out_prompt.verbose_input.text_storage.resize(out_prompt.tokens.size());
        for (size_t i = 0; i < out_prompt.tokens.size(); ++i) {
            out_prompt.verbose_input.text_storage[i] = out_prompt.tokens[i] < 0 ? "<image>" :
                TokenCodec::token_to_piece(model->vocab, out_prompt.tokens[i], true);
        }
        out_prompt.verbose_input.update_ptrs();
    }

    return Status::ok();
}

} // namespace inferlet

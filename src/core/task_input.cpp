#include "task_input.h"
#include "support/validation.h"
#include <cstring>
#include <cmath>
#include <unordered_set>

namespace inferlet {

Status TaskInputInternal::set_prompt_text(const char *text) {
    if (!text) {
        return Status::invalid_argument("prompt_text cannot be NULL");
    }
    if (!is_valid_utf8(text)) {
        return Status::invalid_argument("Invalid UTF-8 in prompt_text");
    }
    // Deep copy into PromptTextInput variant (last-wins)
    input = PromptTextInput{std::string(text)};
    return Status::ok();
}

Status TaskInputInternal::set_prompt_tokens(const int *tokens, int length) {
    if (length < 0) {
        return Status::invalid_argument("prompt_token_length cannot be negative");
    }
    if (length > 0 && !tokens) {
        return Status::invalid_argument("prompt_tokens cannot be NULL when length > 0");
    }

    PromptTokensInput pti;
    pti.tokens.reserve(length);
    for (int i = 0; i < length; ++i) {
        pti.tokens.push_back(tokens[i]);
    }
    input = std::move(pti);
    return Status::ok();
}

Status TaskInputInternal::set_chat_history(const inferlet_chat_history history[], int length) {
    if (length < 0) {
        return Status::invalid_argument("history_length cannot be negative");
    }
    if (length > 0 && !history) {
        return Status::invalid_argument("chat_history cannot be NULL when length > 0");
    }

    ChatInput ci;
    ci.history.reserve(length);
    for (int i = 0; i < length; ++i) {
        ChatMessage msg;
        if (history[i].system) {
            if (!is_valid_utf8(history[i].system)) return Status::invalid_argument("Invalid UTF-8 in system message");
            msg.system = history[i].system;
        }
        if (history[i].query) {
            if (!is_valid_utf8(history[i].query)) return Status::invalid_argument("Invalid UTF-8 in query message");
            msg.query = history[i].query;
        }
        if (history[i].response) {
            if (!is_valid_utf8(history[i].response)) return Status::invalid_argument("Invalid UTF-8 in response message");
            msg.response = history[i].response;
        }
        ci.history.push_back(std::move(msg));
    }
    input = std::move(ci);
    return Status::ok();
}

Status TaskInputInternal::set_multimodal(const inferlet_multimodal mm[], int length) {
    if (length < 0) {
        return Status::invalid_argument("mm_length cannot be negative");
    }
    if (length > 0 && !mm) {
        return Status::invalid_argument("mm cannot be NULL when length > 0");
    }

    LegacyMultiModalInput lmi;
    lmi.items.reserve(length);
    for (int i = 0; i < length; ++i) {
        for (int r = 0; r < 2; ++r) {
            if (mm[i].reserved_p[r] != nullptr) {
                return Status::invalid_argument("MultiModal reserved_p must be NULL");
            }
        }
        for (int r = 0; r < 7; ++r) {
            if (mm[i].reserved_i[r] != 0) {
                return Status::invalid_argument("MultiModal reserved_i must be 0");
            }
        }
        LegacyMultiModalItem item;
        if (mm[i].prompt_text) {
            if (!is_valid_utf8(mm[i].prompt_text)) return Status::invalid_argument("Invalid UTF-8 in multimodal prompt_text");
            item.prompt_text = mm[i].prompt_text;
        }
        if (mm[i].feature) {
            if (mm[i].size <= 0) {
                return Status::invalid_argument("multimodal feature size must be > 0 when feature is non-null");
            }
            item.size = mm[i].size;
            item.feature.resize(mm[i].size);
            std::memcpy(item.feature.data(), mm[i].feature, mm[i].size);
        }
        lmi.items.push_back(std::move(item));
    }
    input = std::move(lmi);
    return Status::ok();
}

Status TaskInputInternal::set_multimodal_chat(const inferlet_multimodal_chat_history history[], int length) {
    if (length < 0) {
        return Status::invalid_argument("history_length cannot be negative");
    }
    if (length > 0 && !history) {
        return Status::invalid_argument("chat_history cannot be NULL when length > 0");
    }

    MultiModalChatInput mmci;
    mmci.history.reserve(length);
    int common_dtype = -1;

    for (int i = 0; i < length; ++i) {
        if (!history[i].role) {
            return Status::invalid_argument("multimodal chat history role cannot be NULL");
        }
        if (history[i].reserved_i != 0) {
            return Status::invalid_argument("MultiModal_ChatHistory reserved_i must be 0");
        }
        MultiModalChatMessage msg;
        msg.role = history[i].role;

        int content_len = history[i].length;
        if (content_len < 0) return Status::invalid_argument("content length cannot be negative");
        if (content_len > 0 && !history[i].content) return Status::invalid_argument("content cannot be NULL when length > 0");

        for (int c = 0; c < content_len; ++c) {
            const auto &src = history[i].content[c];
            if (!src.type) return Status::invalid_argument("content type cannot be NULL");

            MultiModalChatContent item;
            item.type = src.type;
            if (src.cache_key) item.cache_key = src.cache_key;
            item.size = src.size;
            item.grid_t = src.grid_t;
            item.grid_h = src.grid_h;
            item.grid_w = src.grid_w;
            item.data_type = src.data_type;
            item.second_per_grid_t = src.second_per_grid_t;
            item.cache_times = src.cache_times;
            item.fps = (src.fps <= 0.0f) ? 24.0f : src.fps;

            if (item.type == "text") {
                if (src.size != 0 || src.grid_t != 0 || src.grid_h != 0 || src.grid_w != 0 || src.data_type != 0) {
                    return Status::invalid_argument("ChatContent type='text' must have size, grid, and data_type set to 0");
                }
                if (src.data) {
                    const char *text_ptr = reinterpret_cast<const char *>(src.data);
                    if (!is_valid_utf8(text_ptr)) return Status::invalid_argument("Invalid UTF-8 in text chat content");
                    item.text_data = text_ptr;
                }
            } else if (item.type == "image" || item.type == "video") {
                if (!src.data || src.size <= 0) {
                    return Status::invalid_argument("image/video data cannot be NULL and size must be > 0");
                }
                if (src.data_type < 1 || src.data_type > 4) {
                    return Status::invalid_argument("multimodal data_type must be in 1..4");
                }
                if (common_dtype == -1) {
                    common_dtype = src.data_type;
                } else if (common_dtype != src.data_type) {
                    return Status::invalid_argument("multimodal data_type must be consistent across entire task");
                }
                if (src.grid_t <= 0 || src.grid_h <= 0 || src.grid_w <= 0) {
                    return Status::invalid_argument("grid_t, grid_h, grid_w must be > 0");
                }
                if (!item.cache_key.empty() && item.cache_times < 2) {
                    return Status::invalid_argument("cache_times must be >= 2 when cache_key is non-empty");
                }

                item.binary_data.resize(src.size);
                std::memcpy(item.binary_data.data(), src.data, src.size);
            }

            // Copy ext if present
            if (src.ext) {
                for (int r = 0; r < 6; ++r) {
                    if (src.ext->reserved_p[r] != nullptr) {
                        return Status::invalid_argument("ChatContentExt reserved_p must be NULL");
                    }
                }
                for (int r = 0; r < 9; ++r) {
                    if (src.ext->reserved_i[r] != 0) {
                        return Status::invalid_argument("ChatContentExt reserved_i must be 0");
                    }
                }
                if (src.ext->frame_num < 0) {
                    return Status::invalid_argument("frame_num cannot be negative");
                }
                item.ext.frame_num = src.ext->frame_num;
                item.ext.packing_num = (src.ext->packing_num == 0) ? 1 : src.ext->packing_num;
                if (item.ext.packing_num < 1 || item.ext.packing_num > 6) {
                    return Status::invalid_argument("packing_num must be between 1 and 6");
                }
                item.ext.slice_num = src.ext->slice_num;
                if (src.ext->language) item.ext.language = src.ext->language;

                if (src.ext->frame_num > 0) {
                    if (src.ext->frame_index) {
                        item.ext.frame_index.assign(src.ext->frame_index, src.ext->frame_index + src.ext->frame_num);
                    } else {
                        // Generate sequential frame indices
                        item.ext.frame_index.resize(src.ext->frame_num);
                        for (int f = 0; f < src.ext->frame_num; ++f) item.ext.frame_index[f] = f;
                    }
                }
            }

            msg.contents.push_back(std::move(item));
        }
        mmci.history.push_back(std::move(msg));
    }

    input = std::move(mmci);
    return Status::ok();
}

Status TaskInputInternal::set_beam_size(int size) {
    if (size < 1) return Status::invalid_argument("beam_size must be >= 1");
    sampling.beam_size = size;
    return Status::ok();
}

Status TaskInputInternal::set_top_k(int k) {
    if (k < 0) return Status::invalid_argument("top_k must be >= 0");
    sampling.top_k = (k == 0) ? 40 : k; // 0 resets to default 40
    sampling.custom_top_k = (k > 0);
    return Status::ok();
}

Status TaskInputInternal::set_top_p(float p) {
    if (!std::isfinite(p) || p <= 0.0f || p > 1.0f) return Status::invalid_argument("top_p must be finite and in (0, 1]");
    sampling.top_p = p;
    sampling.custom_top_p = true;
    return Status::ok();
}

Status TaskInputInternal::set_temperature(float temp) {
    if (!std::isfinite(temp) || temp < 0.0f) return Status::invalid_argument("temperature must be finite and >= 0.0");
    sampling.temperature = temp;
    sampling.custom_temperature = true;
    return Status::ok();
}

Status TaskInputInternal::set_repetition_penalty(float penalty) {
    if (!std::isfinite(penalty) || penalty <= 0.0f) return Status::invalid_argument("repetition_penalty must be finite and > 0.0");
    sampling.repetition_penalty = penalty;
    return Status::ok();
}

Status TaskInputInternal::set_max_output_tokens(int max_tokens) {
    if (max_tokens <= 0) return Status::invalid_argument("max_output_tokens must be > 0");
    sampling.max_output_tokens = max_tokens;
    return Status::ok();
}

Status TaskInputInternal::set_probability_type(int type) {
    if (type < 0 || type > 2) return Status::invalid_argument("probability_type must be 0, 1, or 2");
    sampling.probability_type = type;
    return Status::ok();
}

Status TaskInputInternal::set_verbose_flag(int flag) {
    if (flag == INFERLET_VERBOSE_F_ALL) {
        output.verbose_flags = (1 << 0) | (1 << 1) | (1 << 2);
        return Status::ok();
    }
    if ((flag & ~7) != 0) {
        return Status::invalid_argument("verbose_flag contains unknown bits");
    }
    output.verbose_flags = flag;
    return Status::ok();
}

Status TaskInputInternal::set_stream_block(int block) {
    if (block != 0 && block != 1) return Status::invalid_argument("stream_block must be 0 or 1");
    output.stream_block = (block != 0);
    return Status::ok();
}

Status TaskInputInternal::set_thinking(int thinking) {
    if (thinking != 0 && thinking != 1) return Status::invalid_argument("thinking must be 0 or 1");
    output.thinking = (thinking != 0);
    return Status::ok();
}

Status TaskInputInternal::set_special_token(int keep_special) {
    if (keep_special != 0 && keep_special != 1) return Status::invalid_argument("keep_special_token must be 0 or 1");
    output.keep_special_tokens = (keep_special != 0);
    return Status::ok();
}

Status TaskInputInternal::set_guided_choices(const char *choices[], int num) {
    if (num < 0) return Status::invalid_argument("guided_choices_num cannot be negative");
    if (num > 0 && !choices) return Status::invalid_argument("guided_choices cannot be NULL when num > 0");

    std::vector<std::string> new_choices;
    new_choices.reserve(num);
    std::unordered_set<std::string> seen;

    for (int i = 0; i < num; ++i) {
        if (!choices[i]) return Status::invalid_argument("guided choice element cannot be NULL");
        if (!is_valid_utf8(choices[i])) return Status::invalid_argument("Invalid UTF-8 in guided choice");
        std::string s(choices[i]);
        if (s.empty()) return Status::invalid_argument("guided choice cannot be empty");
        if (!seen.insert(s).second) return Status::invalid_argument("Duplicate guided choice: " + s);
        new_choices.push_back(std::move(s));
    }
    guided_choices = std::move(new_choices);
    return Status::ok();
}

Status TaskInputInternal::set_tools(const char *tool_strings[], int num) {
    if (num < 0) return Status::invalid_argument("tools_num cannot be negative");
    if (num > 0 && !tool_strings) return Status::invalid_argument("tools cannot be NULL when num > 0");

    std::vector<nlohmann::json> new_tools;
    new_tools.reserve(num);
    for (int i = 0; i < num; ++i) {
        if (!tool_strings[i]) return Status::invalid_argument("tool element cannot be NULL");
        try {
            auto j = nlohmann::json::parse(tool_strings[i]);
            if (!j.is_object()) return Status::invalid_argument("tool element must be a JSON object");
            new_tools.push_back(std::move(j));
        } catch (const std::exception &e) {
            return Status::invalid_argument(std::string("tool element is invalid JSON: ") + e.what());
        }
    }
    tools = std::move(new_tools);
    return Status::ok();
}

Status TaskInputInternal::set_visual_encoder(const char *id) {
    visual_encoder_id = id ? std::string(id) : "";
    return Status::ok();
}

Status TaskInputInternal::create_request(TaskRequest &out_req) const {
    if (std::holds_alternative<std::monostate>(input)) {
        return Status::invalid_argument("No input content was set on TaskInput");
    }
    out_req.input = input;
    out_req.sampling = sampling;
    out_req.output = output;
    out_req.lora_id = lora_id;
    out_req.guided_choices = guided_choices;
    out_req.tools = tools;
    out_req.visual_encoder_id = visual_encoder_id;
    return Status::ok();
}

} // namespace inferlet

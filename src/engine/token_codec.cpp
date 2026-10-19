#include "token_codec.h"
#include <llama.h>
#include <cstring>
#include <algorithm>

namespace inferlet {

size_t TokenCodec::truncate_utf8_boundary(std::string_view text, size_t max_bytes) {
    if (text.size() <= max_bytes) {
        return text.size();
    }
    size_t len = max_bytes;
    const auto *bytes = reinterpret_cast<const unsigned char *>(text.data());

    // Step back over UTF-8 continuation bytes (10xxxxxx)
    while (len > 0 && (bytes[len] & 0xC0) == 0x80) {
        len--;
    }

    // Now bytes[len] is either start of a multibyte sequence or ASCII
    if (len > 0) {
        unsigned char lead = bytes[len - 1];
        if ((lead & 0xE0) == 0xC0) {
            // Needed 2 bytes, only had 1
            len -= 1;
        } else if ((lead & 0xF0) == 0xE0) {
            // Needed 3 bytes, only had 1 or 2
            len -= 1;
        } else if ((lead & 0xF8) == 0xF0) {
            // Needed 4 bytes
            len -= 1;
        }
    }
    return len;
}

Status TokenCodec::tokenize(const llama_vocab *vocab,
                            std::string_view text,
                            int *out_tokens,
                            int max_token_length,
                            int &real_token_length,
                            bool add_special,
                            bool parse_special) {
    if (!vocab) {
        return Status::invalid_handle("Model vocab is NULL");
    }

    // First query required token count with NULL buffer
    int32_t n = llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()),
                               nullptr, 0, add_special, parse_special);
    int required = (n < 0) ? -n : n;
    real_token_length = required;

    std::vector<llama_token> temp_tokens(required);
    if (required > 0) {
        int32_t actual = llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()),
                                        temp_tokens.data(), required, add_special, parse_special);
        if (actual < 0) {
            return Status::inference_error("Tokenization failed during buffer fill");
        }
    }

    if (max_token_length > 0 && out_tokens) {
        int to_copy = std::min(required, max_token_length);
        for (int i = 0; i < to_copy; ++i) {
            out_tokens[i] = temp_tokens[i];
        }
    }

    if (required > max_token_length) {
        return Status::buffer_too_small("Output token buffer is too small");
    }
    return Status::ok();
}

Status TokenCodec::tokenize(const llama_vocab *vocab,
                            std::string_view text,
                            bool add_special,
                            bool parse_special,
                            std::vector<int> &out_tokens) {
    if (!vocab) {
        return Status::invalid_handle("Model vocab is NULL");
    }
    int32_t n = llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()),
                               nullptr, 0, add_special, parse_special);
    int required = (n < 0) ? -n : n;
    out_tokens.resize(required);
    if (required == 0) {
        return Status::ok();
    }

    std::vector<llama_token> temp_tokens(required);
    int32_t actual = llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()),
                                    temp_tokens.data(), required, add_special, parse_special);
    if (actual < 0) {
        out_tokens.clear();
        return Status::inference_error("Tokenization failed during buffer fill");
    }
    for (int i = 0; i < required; ++i) {
        out_tokens[i] = temp_tokens[i];
    }
    return Status::ok();
}


std::string TokenCodec::token_to_piece(const llama_vocab *vocab, int token, bool render_special) {
    if (!vocab) return "";

    char buf[128];
    int32_t n = llama_token_to_piece(vocab, static_cast<llama_token>(token), buf, sizeof(buf), 0, render_special);
    if (n >= 0) {
        return std::string(buf, n);
    }
    // Buffer too small, allocate exact size
    int needed = -n;
    std::string result(needed, '\0');
    llama_token_to_piece(vocab, static_cast<llama_token>(token), result.data(), needed, 0, render_special);
    return result;
}

Status TokenCodec::detokenize(const llama_vocab *vocab,
                              const int *tokens,
                              int token_length,
                              char *out_text,
                              int max_text_length,
                              int &real_text_length,
                              bool skip_special) {
    real_text_length = 0;
    if (!vocab) {
        return Status::invalid_handle("Model vocab is NULL");
    }

    // Validate the entire sequence before decoding or touching the output.
    // llama_token_to_piece throws for IDs outside the vocabulary.
    const int vocab_size = llama_vocab_n_tokens(vocab);
    for (int i = 0; i < token_length; ++i) {
        if (tokens[i] < 0 || tokens[i] >= vocab_size) {
            return Status::invalid_argument("Token ID out of vocab range: " + std::to_string(tokens[i]));
        }
    }

    std::string full_text;
    bool render_special = !skip_special;

    for (int i = 0; i < token_length; ++i) {
        full_text += token_to_piece(vocab, tokens[i], render_special);
    }

    real_text_length = static_cast<int>(full_text.size());

    if (max_text_length <= 0 || !out_text) {
        return (full_text.empty()) ? Status::ok() : Status::buffer_too_small("max_text_length is 0");
    }

    if (full_text.size() < static_cast<size_t>(max_text_length)) {
        std::memcpy(out_text, full_text.data(), full_text.size());
        out_text[full_text.size()] = '\0';
        return Status::ok();
    }

    // Capacity is smaller than full text + '\0'
    size_t allowed_chars = static_cast<size_t>(max_text_length - 1);
    size_t safe_len = truncate_utf8_boundary(full_text, allowed_chars);
    std::memcpy(out_text, full_text.data(), safe_len);
    out_text[safe_len] = '\0';

    return Status::buffer_too_small("Output text buffer is too small; text was safely truncated");
}

} // namespace inferlet

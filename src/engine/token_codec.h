#ifndef INFERLET_ENGINE_TOKEN_CODEC_H_INCLUDED
#define INFERLET_ENGINE_TOKEN_CODEC_H_INCLUDED

#include "support/status.h"
#include <string>
#include <vector>
#include <string_view>

struct llama_vocab;

namespace inferlet {

class TokenCodec {
public:
    static Status tokenize(const llama_vocab *vocab,
                           std::string_view text,
                           int *out_tokens,
                           int max_token_length,
                           int &real_token_length,
                           bool add_special,
                           bool parse_special);

    static Status tokenize(const llama_vocab *vocab,
                           std::string_view text,
                           bool add_special,
                           bool parse_special,
                           std::vector<int> &out_tokens);

    static Status detokenize(const llama_vocab *vocab,
                             const int *tokens,
                             int token_length,
                             char *out_text,
                             int max_text_length,
                             int &real_text_length,
                             bool skip_special);

    static std::string token_to_piece(const llama_vocab *vocab, int token, bool render_special);

    static size_t truncate_utf8_boundary(std::string_view text, size_t max_bytes);
};

} // namespace inferlet

#endif // INFERLET_ENGINE_TOKEN_CODEC_H_INCLUDED

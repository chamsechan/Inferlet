#include "inferlet.h"
#include "support/validation.h"
#include "core/model_handle.h"
#include "engine/llama_engine.h"
#include <cassert>
#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <cstring>
#include <limits>

using namespace inferlet;

const char *get_model_path() {
    const char *env = std::getenv("INFERLET_TEST_MODEL");
    if (env && std::strlen(env) > 0) {
        return env;
    }
    return "/home/ubuntu/project/llm-ops-agy/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
}

void test_tokenizer_roundtrip() {
    inferlet_init();

    void *param = inferlet_parameter_allocate();
    inferlet_parameter_set_tokenizer_only(param, 1);
    inferlet_parameter_set_load_from_file_sync(param, 1);

    const char *model_path = get_model_path();
    void *handle = inferlet_load_from_file(model_path, param);
    assert(handle != nullptr && "Tokenizer integration requires a loadable test model");

    const std::string original_text = "Hello, Inferlet! Testing tokenization and detokenization 12345.";

    int tokens[256];
    int real_tokens = 0;
    int ret = inferlet_tokenizer_encode(handle, original_text.c_str(), -1, tokens, 256, &real_tokens, 0, 0);
    assert(ret == INFERLET_OK);
    assert(real_tokens > 0);

    char decoded_text[512];
    int real_text_len = 0;
    ret = inferlet_tokenizer_decode(handle, tokens, real_tokens, decoded_text, sizeof(decoded_text), &real_text_len, 1);
    assert(ret == INFERLET_OK);
    assert(real_text_len == static_cast<int>(original_text.size()));
    assert(original_text == std::string(decoded_text));

    std::cout << "[PASS] test_tokenizer_roundtrip\n";

    const int vocab_size = LlamaEngine::get_vocab_n_tokens(static_cast<ModelHandleInternal *>(handle)->vocab());
    for (int bad_token : {-1, vocab_size, std::numeric_limits<int>::max()}) {
        for (int skip_special : {0, 1}) {
            int invalid_tokens[] = {tokens[0], bad_token};
            char untouched[] = "unchanged";
            int length = 123;
            assert(inferlet_tokenizer_decode(handle, invalid_tokens, 2, untouched, sizeof(untouched),
                                           &length, skip_special) == INFERLET_E_INVALID_ARGUMENT);
            assert(length == 0);
            assert(std::strcmp(untouched, "unchanged") == 0);
            length = 123;
            assert(inferlet_tokenizer_decode(handle, invalid_tokens, 2, nullptr, 0,
                                           &length, skip_special) == INFERLET_E_INVALID_ARGUMENT);
            assert(length == 0);
        }
    }
    const int last_token = vocab_size - 1;
    int last_token_length = 0;
    assert(inferlet_tokenizer_decode(handle, &last_token, 1, decoded_text, sizeof(decoded_text),
                                   &last_token_length, 1) == INFERLET_OK);
    std::cout << "[PASS] test_tokenizer_invalid_token_ids\n";

    void *generation_input = inferlet_task_input_allocate();
    assert(generation_input != nullptr);
    assert(inferlet_task_input_set_prompt_text(generation_input, "This must not decode") == INFERLET_OK);
    void *generation_output = nullptr;
    ret = inferlet_run(handle, generation_input, &generation_output);
    assert(ret == INFERLET_E_UNSUPPORTED);
    assert(generation_output == nullptr);
    inferlet_task_input_deallocate(generation_input);
    std::cout << "[PASS] test_tokenizer_only_generation_rejected\n";

    // Test buffer too small for Encode
    int tiny_tokens[3];
    int tiny_real_tokens = 0;
    ret = inferlet_tokenizer_encode(handle, original_text.c_str(), -1, tiny_tokens, 3, &tiny_real_tokens, 0, 0);
    assert(ret == INFERLET_E_BUFFER_TOO_SMALL);
    assert(tiny_real_tokens == real_tokens);
    assert(tiny_tokens[0] == tokens[0]);
    assert(tiny_tokens[1] == tokens[1]);
    assert(tiny_tokens[2] == tokens[2]);

    std::cout << "[PASS] test_tokenizer_encode_buffer_too_small\n";

    // Test buffer too small for Decode
    char small_text[10];
    int small_real_text = 0;
    ret = inferlet_tokenizer_decode(handle, tokens, real_tokens, small_text, 10, &small_real_text, 1);
    assert(ret == INFERLET_E_BUFFER_TOO_SMALL);
    assert(small_real_text == real_text_len);
    assert(is_valid_utf8(small_text));
    assert(std::strlen(small_text) < 10);

    std::cout << "[PASS] test_tokenizer_decode_buffer_too_small\n";

    // Test Chinese UTF-8 boundary safety
    const std::string chinese_text = "你好世界，人工智能大模型！";
    int cn_tokens[256];
    int cn_real_tokens = 0;
    ret = inferlet_tokenizer_encode(handle, chinese_text.c_str(), -1, cn_tokens, 256, &cn_real_tokens, 0, 0);
    assert(ret == INFERLET_OK);
    assert(cn_real_tokens > 0);

    char cn_decoded[512];
    int cn_real_text = 0;
    ret = inferlet_tokenizer_decode(handle, cn_tokens, cn_real_tokens, cn_decoded, sizeof(cn_decoded), &cn_real_text, 1);
    assert(ret == INFERLET_OK);
    assert(chinese_text == std::string(cn_decoded));

    // Decode with buffer boundary cutting through a 3-byte Chinese character
    char cn_short[8]; // allows 7 bytes max
    int cn_short_real = 0;
    ret = inferlet_tokenizer_decode(handle, cn_tokens, cn_real_tokens, cn_short, 8, &cn_short_real, 1);
    assert(ret == INFERLET_E_BUFFER_TOO_SMALL);
    assert(cn_short_real == cn_real_text);
    assert(is_valid_utf8(cn_short));
    // 2 Chinese chars = 6 bytes. 3rd would be 9 bytes. So cn_short should be exactly 6 bytes of text + '\0'
    assert(std::strlen(cn_short) == 6);

    std::cout << "[PASS] test_tokenizer_utf8_boundary_safety\n";

    // Test special token parsing
    const std::string special_text = "<|im_start|>user\nHello<|im_end|>";
    int parse_special_tokens[256];
    int parse_real = 0;
    ret = inferlet_tokenizer_encode(handle, special_text.c_str(), -1, parse_special_tokens, 256, &parse_real, 0, 1);
    assert(ret == INFERLET_OK);

    int raw_tokens[256];
    int raw_real = 0;
    ret = inferlet_tokenizer_encode(handle, special_text.c_str(), -1, raw_tokens, 256, &raw_real, 0, 0);
    assert(ret == INFERLET_OK);
    // With parse_special = 1, special tokens are recognized as single control tokens, so fewer tokens
    assert(parse_real < raw_real);

    std::cout << "[PASS] test_tokenizer_special_tokens\n";

    // Model sharing test
    void *handle2 = inferlet_load_from_file(model_path, param);
    assert(handle2 != nullptr);

    // Conflicting configuration should fail
    void *param_conflict = inferlet_parameter_allocate();
    inferlet_parameter_set_tokenizer_only(param_conflict, 0); // conflicting
    void *handle_conflict = inferlet_load_from_file(model_path, param_conflict);
    assert(handle_conflict == nullptr);
    inferlet_parameter_deallocate(param_conflict);

    std::cout << "[PASS] test_model_registry_sharing\n";

    inferlet_unload(handle2);
    inferlet_unload(handle);
    inferlet_parameter_deallocate(param);
    inferlet_deinit();
}

int main() {
    test_tokenizer_roundtrip();
    std::cout << "All Tokenizer and ModelRegistry integration tests passed successfully!\n";
    return 0;
}

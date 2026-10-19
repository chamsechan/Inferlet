#include "inferlet.h"
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <cstdlib>

static const char *get_default_model() {
    const char *env = std::getenv("INFERLET_TEST_MODEL");
    if (env && std::strlen(env) > 0) {
        return env;
    }
    return "/home/ubuntu/project/llm-ops-agy/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
}

int main(int argc, char **argv) {
    const char *model_path = get_default_model();
    const char *text = "Hello world! Testing Inferlet tokenizer encoding and decoding.";

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (std::strcmp(argv[i], "--text") == 0 && i + 1 < argc) {
            text = argv[++i];
        }
    }

    std::cout << "[example_tokenizer] Initializing Inferlet...\n";
    inferlet_init();

    void *param = inferlet_parameter_allocate();
    inferlet_parameter_set_tokenizer_only(param, 1); // Fast vocab-only mode
    inferlet_parameter_set_load_from_file_sync(param, 1);

    std::cout << "[example_tokenizer] Loading tokenizer vocab from: " << model_path << "\n";
    void *handle = inferlet_load_from_file(model_path, param);
    inferlet_parameter_deallocate(param);

    if (!handle) {
        std::cerr << "[example_tokenizer] Failed to load tokenizer vocab: " << model_path << "\n";
        inferlet_deinit();
        return 0;
    }

    std::cout << "[example_tokenizer] Input text: \"" << text << "\"\n";

    // 1. Encode text to tokens
    std::vector<int> tokens(256, 0);
    int real_token_cnt = 0;
    int text_len = static_cast<int>(std::strlen(text));
    int ret = inferlet_tokenizer_encode(handle, text, text_len, tokens.data(), static_cast<int>(tokens.size()), &real_token_cnt, 1, 0);

    if (ret != INFERLET_OK) {
        std::cerr << "[example_tokenizer] Tokenizer encode failed: " << ret << "\n";
        inferlet_unload(handle);
        inferlet_deinit();
        return 1;
    }

    std::cout << "[example_tokenizer] Encoded " << real_token_cnt << " tokens:\n  [";
    for (int i = 0; i < real_token_cnt; ++i) {
        std::cout << tokens[i] << (i + 1 < real_token_cnt ? ", " : "");
    }
    std::cout << "]\n";

    // 2. Decode tokens back to text
    std::vector<char> decoded_buf(512, '\0');
    int real_text_len = 0;
    ret = inferlet_tokenizer_decode(handle, tokens.data(), real_token_cnt, decoded_buf.data(), static_cast<int>(decoded_buf.size()), &real_text_len, 0);


    if (ret != INFERLET_OK) {
        std::cerr << "[example_tokenizer] Tokenizer decode failed: " << ret << "\n";
    } else {
        std::cout << "[example_tokenizer] Decoded text (" << real_text_len << " bytes):\n  \"" << decoded_buf.data() << "\"\n";
    }

    inferlet_unload(handle);
    inferlet_deinit();
    std::cout << "[example_tokenizer] Finished successfully.\n";
    return 0;
}

#include "inferlet.h"
#include "engine/prompt_builder.h"
#include "engine/llama_engine.h"
#include "core/runtime.h"
#include "core/model_registry.h"
#include <cassert>
#include <cstring>
#include <iostream>

using namespace inferlet;

const char *get_test_model_path() {
    const char *env = std::getenv("INFERLET_TEST_MODEL");
    if (env && std::strlen(env) > 0) return env;
    return "/home/ubuntu/project/llm-ops-agy/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
}

void test_unfold_chat_history_rules() {
    std::vector<std::pair<std::string, std::string>> msgs;

    // 1. Empty history
    {
        std::vector<ChatMessage> history;
        Status st = PromptBuilder::unfold_chat_history(history, msgs);
        assert(!st.is_ok());
        assert(st.code == INFERLET_E_INVALID_ARGUMENT);
    }

    // 2. Conflicting system prompts
    {
        std::vector<ChatMessage> history = {
            {"System 1", "Hello", "Hi"},
            {"System 2", "Next", ""}
        };
        Status st = PromptBuilder::unfold_chat_history(history, msgs);
        assert(!st.is_ok());
        assert(st.code == INFERLET_E_INVALID_ARGUMENT);
    }

    // 3. Identical repeated system prompts should be accepted (ignored subsequent)
    {
        std::vector<ChatMessage> history = {
            {"System A", "Hello", "Hi"},
            {"System A", "Next query", ""}
        };
        Status st = PromptBuilder::unfold_chat_history(history, msgs);
        assert(st.is_ok());
        assert(msgs.size() == 4);
        assert(msgs[0].first == "system" && msgs[0].second == "System A");
        assert(msgs[1].first == "user" && msgs[1].second == "Hello");
        assert(msgs[2].first == "assistant" && msgs[2].second == "Hi");
        assert(msgs[3].first == "user" && msgs[3].second == "Next query");
    }

    // 4. Ending with assistant response without continuation must fail
    {
        std::vector<ChatMessage> history = {
            {"", "What is 1+1?", "2"}
        };
        Status st = PromptBuilder::unfold_chat_history(history, msgs);
        assert(!st.is_ok());
        assert(st.code == INFERLET_E_INVALID_ARGUMENT);
    }

    // 5. Valid single turn ending with user query
    {
        std::vector<ChatMessage> history = {
            {"You are AI", "Tell me a joke", ""}
        };
        Status st = PromptBuilder::unfold_chat_history(history, msgs);
        assert(st.is_ok());
        assert(msgs.size() == 2);
        assert(msgs[0].first == "system" && msgs[0].second == "You are AI");
        assert(msgs[1].first == "user" && msgs[1].second == "Tell me a joke");
    }

    std::cout << "[PASS] test_unfold_chat_history_rules\n";
}

void test_chat_template_golden_and_no_template() {
    // 1. Test invalid / no template fails with unsupported
    {
        std::vector<ChatMessage> history = {{"", "Hi", ""}};
        std::string formatted;
        Status st = PromptBuilder::apply_chat_template(history, "invalid_non_existent_template_xyz", nullptr, formatted);
        assert(!st.is_ok());
    }

    // 2. Test built-in chatml template with custom string
    {
        std::vector<ChatMessage> history = {
            {"You are a helpful assistant.", "Hello", "Hi!"},
            {"", "How are you?", ""}
        };
        std::string formatted;
        Status st = PromptBuilder::apply_chat_template(history, "chatml", nullptr, formatted);
        assert(st.is_ok());
        assert(formatted.find("<|im_start|>system\nYou are a helpful assistant.<|im_end|>") != std::string::npos);
        assert(formatted.find("<|im_start|>user\nHello<|im_end|>") != std::string::npos);
        assert(formatted.find("<|im_start|>assistant\nHi!<|im_end|>") != std::string::npos);
        assert(formatted.find("<|im_start|>user\nHow are you?<|im_end|>") != std::string::npos);
        assert(formatted.find("<|im_start|>assistant\n") != std::string::npos);
    }

    // 3. Test built-in llama2 template with custom string
    {
        std::vector<ChatMessage> history = {
            {"Be concise.", "What is Paris?", ""}
        };
        std::string formatted;
        Status st = PromptBuilder::apply_chat_template(history, "llama2", nullptr, formatted);
        assert(st.is_ok());
        assert(formatted.find("What is Paris?") != std::string::npos);
    }

    std::cout << "[PASS] test_chat_template_golden_and_no_template\n";
}

void test_prompt_builder_with_model() {
    inferlet_init();

    const char *model_path = get_test_model_path();
    ModelConfig cfg;
    cfg.tokenizer_only = true;

    std::shared_ptr<SharedModel> model;
    Status st = ModelRegistry::instance().get_or_load(model_path, cfg, model);
    if (!st.is_ok() || !model) {
        std::cout << "Could not load test model, skipping model-dependent tests\n";
        inferlet_deinit();
        return;
    }
    {
        std::unique_lock<std::mutex> lock(model->mutex);
        if (model->load_state == LoadState::Loading) {
            model->load_cv.wait(lock, [&model]() {
                return model->load_state != LoadState::Loading;
            });
        }
        if (model->load_state != LoadState::Ready) {
            std::cout << "Could not load test model, skipping model-dependent tests\n";
            inferlet_deinit();
            return;
        }
    }

    // 1. Text input prompt building
    {
        TaskRequest req;
        req.input = PromptTextInput{"Hello world"};
        req.output.verbose_flags = INFERLET_VERBOSE_F_INPUT_TOKENS;

        PreparedPrompt prompt;
        st = PromptBuilder::build(req, model, prompt);
        assert(st.is_ok());
        assert(!prompt.tokens.empty());
        assert(prompt.verbose_input.tokens.size() == prompt.tokens.size());
        assert(prompt.verbose_input.text_storage.size() == prompt.tokens.size());
        assert(prompt.verbose_input.text_ptrs.size() == prompt.tokens.size());
        for (size_t i = 0; i < prompt.tokens.size(); ++i) {
            assert(prompt.verbose_input.text_ptrs[i] == prompt.verbose_input.text_storage[i].c_str());
            assert(!prompt.verbose_input.text_storage[i].empty());
        }
    }

    // 2. Empty text input resolves to BOS or error
    {
        TaskRequest req;
        req.input = PromptTextInput{""};
        PreparedPrompt prompt;
        st = PromptBuilder::build(req, model, prompt);
        int bos = LlamaEngine::get_vocab_bos(model->vocab);
        if (bos != -1 && !LlamaEngine::is_vocab_eog(model->vocab, bos)) {
            assert(st.is_ok());
            assert(prompt.tokens.size() == 1);
            assert(prompt.tokens[0] == bos);
        } else {
            assert(!st.is_ok());
            assert(st.code == INFERLET_E_INVALID_ARGUMENT);
        }
    }

    // 3. Tokens input prompt building
    {
        TaskRequest req;
        req.input = PromptTokensInput{{10, 20, 30}};
        PreparedPrompt prompt;
        st = PromptBuilder::build(req, model, prompt);
        assert(st.is_ok());
        assert(prompt.tokens.size() == 3);
        assert(prompt.tokens[0] == 10);
        assert(prompt.tokens[1] == 20);
        assert(prompt.tokens[2] == 30);

        // Out of vocab token
        req.input = PromptTokensInput{{99999999}};
        st = PromptBuilder::build(req, model, prompt);
        assert(!st.is_ok());
        assert(st.code == INFERLET_E_INVALID_ARGUMENT);
    }

    // 4. Chat input prompt building with real model template
    {
        TaskRequest req;
        ChatInput ci;
        ci.history.push_back({"You are a helpful assistant.", "What is the capital of France?", ""});
        req.input = ci;
        req.output.verbose_flags = INFERLET_VERBOSE_F_INPUT_TOKENS;

        PreparedPrompt prompt;
        st = PromptBuilder::build(req, model, prompt);
        assert(st.is_ok());
        assert(!prompt.tokens.empty());
        assert(!prompt.formatted_text.empty());
        assert(prompt.formatted_text.find("France") != std::string::npos);
        assert(prompt.verbose_input.tokens.size() == prompt.tokens.size());
    }

    ModelRegistry::instance().release(model);
    inferlet_deinit();
    std::cout << "[PASS] test_prompt_builder_with_model\n";
}

int main() {
    std::cout << "Starting PromptBuilder unit tests...\n";
    test_unfold_chat_history_rules();
    test_chat_template_golden_and_no_template();
    test_prompt_builder_with_model();
    std::cout << "All PromptBuilder tests passed!\n";
    return 0;
}

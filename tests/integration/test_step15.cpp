#include "inferlet.h"
#include <cassert>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <thread>
#include <future>
#include <cstdlib>
#include <fstream>

static const char *TEST_MODEL_PATH = [] {
    const char *path = std::getenv("INFERLET_TEST_MODEL");
    return path && *path ? path : "/home/ubuntu/project/llm-ops-agy/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
}();
static const char *TEST_LORA_PATH = "/tmp/inferlet_test_step15_lora.gguf";

static void create_test_lora_file() {
    std::string cmd = "python3 -c \""
                      "import sys\n"
                      "sys.path.append('third_party/llama.cpp/gguf-py')\n"
                      "sys.path.append('../third_party/llama.cpp/gguf-py')\n"
                      "import gguf\n"
                      "writer = gguf.GGUFWriter('" + std::string(TEST_LORA_PATH) + "', 'qwen2')\n"
                      "writer.add_string('general.type', 'adapter')\n"
                      "writer.add_string('adapter.type', 'lora')\n"
                      "writer.add_float32('adapter.lora.alpha', 1.0)\n"
                      "writer.write_header_to_file()\n"
                      "writer.write_kv_data_to_file()\n"
                      "writer.write_tensors_to_file()\n"
                      "writer.close()\n"
                      "\"";
    int ret = std::system(cmd.c_str());
    assert(ret == 0 && "Failed to generate test LoRA GGUF");
}

static void remove_test_lora_file() {
    std::remove(TEST_LORA_PATH);
}

static void test_guided_choice_setter_validation() {
    std::cout << "[Step15 Test] Running test_guided_choice_setter_validation..." << std::endl;
    void *input = inferlet_task_input_allocate();
    assert(input != nullptr);

    const char *valid_choices[] = {"apple", "banana", "cherry"};
    assert(inferlet_task_input_set_guided_choices(input, valid_choices, 3) == INFERLET_OK);

    // Negative num
    assert(inferlet_task_input_set_guided_choices(input, valid_choices, -1) == INFERLET_E_INVALID_ARGUMENT);

    // NULL array with positive num
    assert(inferlet_task_input_set_guided_choices(input, nullptr, 2) == INFERLET_E_INVALID_ARGUMENT);

    // NULL element
    const char *null_elem[] = {"apple", nullptr, "cherry"};
    assert(inferlet_task_input_set_guided_choices(input, null_elem, 3) == INFERLET_E_INVALID_ARGUMENT);

    // Empty string choice
    const char *empty_str[] = {"apple", "", "cherry"};
    assert(inferlet_task_input_set_guided_choices(input, empty_str, 3) == INFERLET_E_INVALID_ARGUMENT);

    // Duplicate choices
    const char *dups[] = {"apple", "banana", "apple"};
    assert(inferlet_task_input_set_guided_choices(input, dups, 3) == INFERLET_E_INVALID_ARGUMENT);

    // Zero count clears
    assert(inferlet_task_input_set_guided_choices(input, nullptr, 0) == INFERLET_OK);

    inferlet_task_input_deallocate(input);
    std::cout << "  -> PASSED" << std::endl;
}

static void test_tools_setter_validation() {
    std::cout << "[Step15 Test] Running test_tools_setter_validation..." << std::endl;
    void *input = inferlet_task_input_allocate();
    assert(input != nullptr);

    const char *valid_tool = "{\"name\": \"get_weather\", \"description\": \"Get current weather\"}";
    const char *tools[] = {valid_tool};
    assert(inferlet_task_input_set_tools(input, tools, 1) == INFERLET_OK);

    // Negative num
    assert(inferlet_task_input_set_tools(input, tools, -1) == INFERLET_E_INVALID_ARGUMENT);

    // NULL array with positive num
    assert(inferlet_task_input_set_tools(input, nullptr, 1) == INFERLET_E_INVALID_ARGUMENT);

    // NULL element
    const char *null_tool[] = {nullptr};
    assert(inferlet_task_input_set_tools(input, null_tool, 1) == INFERLET_E_INVALID_ARGUMENT);

    // Invalid JSON syntax
    const char *bad_json[] = {"{not a valid json}"};
    assert(inferlet_task_input_set_tools(input, bad_json, 1) == INFERLET_E_INVALID_ARGUMENT);

    // Valid JSON but not an object (JSON array)
    const char *arr_json[] = {"[\"item1\", \"item2\"]"};
    assert(inferlet_task_input_set_tools(input, arr_json, 1) == INFERLET_E_INVALID_ARGUMENT);

    // Valid JSON but scalar
    const char *str_json[] = {"\"just_a_string\""};
    assert(inferlet_task_input_set_tools(input, str_json, 1) == INFERLET_E_INVALID_ARGUMENT);

    // Zero count clears
    assert(inferlet_task_input_set_tools(input, nullptr, 0) == INFERLET_OK);

    inferlet_task_input_deallocate(input);
    std::cout << "  -> PASSED" << std::endl;
}

static void test_thinking_setter_validation() {
    std::cout << "[Step15 Test] Running test_thinking_setter_validation..." << std::endl;
    void *input = inferlet_task_input_allocate();
    assert(input != nullptr);

    assert(inferlet_task_input_set_thinking_mode(input, 0) == INFERLET_OK);
    assert(inferlet_task_input_set_thinking_mode(input, 1) == INFERLET_OK);
    assert(inferlet_task_input_set_thinking_mode(input, -1) == INFERLET_E_INVALID_ARGUMENT);
    assert(inferlet_task_input_set_thinking_mode(input, 2) == INFERLET_E_INVALID_ARGUMENT);

    inferlet_task_input_deallocate(input);
    std::cout << "  -> PASSED" << std::endl;
}

static void test_lora_setter_validation() {
    std::cout << "[Step15 Test] Running test_lora_setter_validation..." << std::endl;
    void *input = inferlet_task_input_allocate();
    assert(input != nullptr);

    assert(inferlet_task_input_set_low_rank_adapter_id(input, 0) == INFERLET_OK);
    assert(inferlet_task_input_set_low_rank_adapter_id(input, 1) == INFERLET_OK);
    assert(inferlet_task_input_set_low_rank_adapter_id(input, 100) == INFERLET_OK);
    assert(inferlet_task_input_set_low_rank_adapter_id(input, -1) == INFERLET_E_INVALID_ARGUMENT);

    inferlet_task_input_deallocate(input);
    std::cout << "  -> PASSED" << std::endl;
}

static void test_guided_choice_execution(void *model) {
    std::cout << "[Step15 Test] Running test_guided_choice_execution..." << std::endl;

    void *input = inferlet_task_input_allocate();
    assert(input != nullptr);
    assert(inferlet_task_input_set_prompt_text(input, "Complete with one word: The color of grass is") == INFERLET_OK);
    assert(inferlet_task_input_set_max_output_tokens(input, 10) == INFERLET_OK);

    const char *choices[] = {"green", "purple", "metallic"};
    assert(inferlet_task_input_set_guided_choices(input, choices, 3) == INFERLET_OK);

    void *output = nullptr;
    assert(inferlet_run(model, input, &output) == INFERLET_OK);
    assert(output != nullptr);

    const char *text = inferlet_task_output_get_result_text(output);
    assert(text != nullptr);
    std::cout << "  -> Guided choices output: '" << text << "'" << std::endl;

    bool matched = (std::strcmp(text, "green") == 0 ||
                    std::strcmp(text, "purple") == 0 ||
                    std::strcmp(text, "metallic") == 0);
    assert(matched && "Output must exactly match one of the guided choices");

    inferlet_task_output_deallocate(output);

    // Test with common prefix
    const char *prefix_choices[] = {"red apple", "red cherry", "green apple"};
    assert(inferlet_task_input_set_prompt_text(input, "Pick a fruit:") == INFERLET_OK);
    assert(inferlet_task_input_set_guided_choices(input, prefix_choices, 3) == INFERLET_OK);

    output = nullptr;
    assert(inferlet_run(model, input, &output) == INFERLET_OK);
    assert(output != nullptr);

    text = inferlet_task_output_get_result_text(output);
    assert(text != nullptr);
    std::cout << "  -> Prefix guided choices output: '" << text << "'" << std::endl;

    matched = (std::strcmp(text, "red apple") == 0 ||
               std::strcmp(text, "red cherry") == 0 ||
               std::strcmp(text, "green apple") == 0);
    assert(matched && "Output must match one of the prefix guided choices");

    inferlet_task_output_deallocate(output);
    inferlet_task_input_deallocate(input);
    std::cout << "  -> PASSED" << std::endl;
}

static void test_guided_choice_and_tools_conflict(void *model) {
    std::cout << "[Step15 Test] Running test_guided_choice_and_tools_conflict..." << std::endl;

    void *input = inferlet_task_input_allocate();
    assert(input != nullptr);
    assert(inferlet_task_input_set_prompt_text(input, "Hello") == INFERLET_OK);

    const char *choices[] = {"yes", "no"};
    assert(inferlet_task_input_set_guided_choices(input, choices, 2) == INFERLET_OK);

    const char *tools[] = {"{\"name\": \"test_func\"}"};
    assert(inferlet_task_input_set_tools(input, tools, 1) == INFERLET_OK);

    void *output = nullptr;
    int rc = inferlet_run(model, input, &output);
    assert(rc == INFERLET_E_UNSUPPORTED && "Simultaneous guided choices and tools must return INFERLET_E_UNSUPPORTED");
    assert(output == nullptr);

    inferlet_task_input_deallocate(input);
    std::cout << "  -> PASSED" << std::endl;
}

static void test_tools_execution(void *model) {
    std::cout << "[Step15 Test] Running test_tools_execution..." << std::endl;

    void *input = inferlet_task_input_allocate();
    assert(input != nullptr);

    inferlet_chat_history hist[] = {
        {nullptr, "What's the weather in Beijing?", nullptr}
    };
    assert(inferlet_task_input_set_chat_history(input, hist, 1) == INFERLET_OK);
    assert(inferlet_task_input_set_max_output_tokens(input, 30) == INFERLET_OK);

    const char *tools[] = {
        "{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"description\": \"Get weather for city\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}}}}}"
    };
    assert(inferlet_task_input_set_tools(input, tools, 1) == INFERLET_OK);

    void *output = nullptr;
    assert(inferlet_run(model, input, &output) == INFERLET_OK);
    assert(output != nullptr);

    const char *res = inferlet_task_output_get_result_text(output);
    assert(res != nullptr);
    std::cout << "  -> Tools prompt generation result: " << res << std::endl;

    inferlet_task_output_deallocate(output);

    // Tools on raw token input must fail with unsupported
    int raw_toks[] = {1, 2, 3};
    assert(inferlet_task_input_set_prompt_tokens(input, raw_toks, 3) == INFERLET_OK);
    output = nullptr;
    int rc = inferlet_run(model, input, &output);
    assert(rc == INFERLET_E_UNSUPPORTED && "Tools on raw token input must return unsupported");

    inferlet_task_input_deallocate(input);
    std::cout << "  -> PASSED" << std::endl;
}

static void test_thinking_mode_execution(void *model) {
    std::cout << "[Step15 Test] Running test_thinking_mode_execution..." << std::endl;

    void *input = inferlet_task_input_allocate();
    assert(input != nullptr);
    assert(inferlet_task_input_set_prompt_text(input, "Say hi") == INFERLET_OK);
    assert(inferlet_task_input_set_max_output_tokens(input, 5) == INFERLET_OK);

    // Thinking mode 0 (non-thinking)
    assert(inferlet_task_input_set_thinking_mode(input, 0) == INFERLET_OK);
    void *out0 = nullptr;
    assert(inferlet_run(model, input, &out0) == INFERLET_OK);
    assert(out0 != nullptr);
    inferlet_task_output_deallocate(out0);

    // Thinking mode 1 (thinking)
    assert(inferlet_task_input_set_thinking_mode(input, 1) == INFERLET_OK);
    void *out1 = nullptr;
    assert(inferlet_run(model, input, &out1) == INFERLET_OK);
    assert(out1 != nullptr);
    inferlet_task_output_deallocate(out1);

    inferlet_task_input_deallocate(input);
    std::cout << "  -> PASSED" << std::endl;
}

static void test_unregistered_lora_id(void *model) {
    std::cout << "[Step15 Test] Running test_unregistered_lora_id..." << std::endl;

    void *input = inferlet_task_input_allocate();
    assert(input != nullptr);
    assert(inferlet_task_input_set_prompt_text(input, "Hello") == INFERLET_OK);
    assert(inferlet_task_input_set_low_rank_adapter_id(input, 999) == INFERLET_OK);

    void *output = nullptr;
    int rc = inferlet_run(model, input, &output);
    assert(rc == INFERLET_E_UNSUPPORTED && "Unregistered LoRA ID must return unsupported");
    assert(output == nullptr);

    inferlet_task_input_deallocate(input);
    std::cout << "  -> PASSED" << std::endl;
}

static void test_invalid_lora_file_load() {
    std::cout << "[Step15 Test] Running test_invalid_lora_file_load..." << std::endl;

    void *param = inferlet_parameter_allocate();
    assert(param != nullptr);
    inferlet_parameter_set_low_rank_adapter(param, 1, "/path/to/nonexistent/lora_adapter.gguf");
    inferlet_parameter_set_load_from_file_sync(param, 1);

    void *bad_model = inferlet_load_from_file(TEST_MODEL_PATH, param);
    assert(bad_model == nullptr && "Loading with invalid LoRA path must fail");

    inferlet_parameter_deallocate(param);
    std::cout << "  -> PASSED" << std::endl;
}

static void test_valid_lora_and_concurrent_isolation() {
    std::cout << "[Step15 Test] Running test_valid_lora_and_concurrent_isolation..." << std::endl;

    create_test_lora_file();

    void *param = inferlet_parameter_allocate();
    assert(param != nullptr);
    inferlet_parameter_set_low_rank_adapter(param, 1, TEST_LORA_PATH);
    inferlet_parameter_set_backend_cnt(param, 2);
    inferlet_parameter_set_load_from_file_sync(param, 1);

    void *model_with_lora = inferlet_load_from_file(TEST_MODEL_PATH, param);
    assert(model_with_lora != nullptr && "Model load with valid LoRA must succeed");

    // Concurrently run Task 1 (lora_id=1) and Task 2 (lora_id=0)
    auto run_task = [](void *m, int lora_id, const std::string &prompt) -> bool {
        void *in = inferlet_task_input_allocate();
        if (!in) return false;
        inferlet_task_input_set_prompt_text(in, prompt.c_str());
        inferlet_task_input_set_max_output_tokens(in, 10);
        inferlet_task_input_set_low_rank_adapter_id(in, lora_id);

        void *out = nullptr;
        int status = inferlet_run(m, in, &out);
        bool ok = (status == INFERLET_OK && out != nullptr);
        if (out) {
            const char *txt = inferlet_task_output_get_result_text(out);
            assert(txt != nullptr);
            inferlet_task_output_deallocate(out);
        }
        inferlet_task_input_deallocate(in);
        return ok;
    };

    auto f1 = std::async(std::launch::async, run_task, model_with_lora, 1, "The capital of France is");
    auto f2 = std::async(std::launch::async, run_task, model_with_lora, 0, "The capital of Germany is");

    assert(f1.get() && "Task with lora_id=1 must succeed");
    assert(f2.get() && "Task with lora_id=0 must succeed");

    inferlet_unload(model_with_lora);
    inferlet_parameter_deallocate(param);
    remove_test_lora_file();

    std::cout << "  -> PASSED" << std::endl;
}

int main() {
    std::cout << "========== Step 15 Guided, Tools, Thinking, and LoRA Tests ==========" << std::endl;
    inferlet_init();

    // Setter validations
    test_guided_choice_setter_validation();
    test_tools_setter_validation();
    test_thinking_setter_validation();
    test_lora_setter_validation();

    // Model load tests with invalid LoRA
    test_invalid_lora_file_load();

    // Load base model for execution tests
    void *param = inferlet_parameter_allocate();
    assert(param != nullptr);
    inferlet_parameter_set_backend_cnt(param, 2);
    inferlet_parameter_set_load_from_file_sync(param, 1);

    void *base_model = inferlet_load_from_file(TEST_MODEL_PATH, param);
    assert(base_model != nullptr && "Failed to load base test model");

    test_guided_choice_execution(base_model);
    test_guided_choice_and_tools_conflict(base_model);
    test_tools_execution(base_model);
    test_thinking_mode_execution(base_model);
    test_unregistered_lora_id(base_model);

    inferlet_unload(base_model);
    inferlet_parameter_deallocate(param);

    // Test valid LoRA loading & concurrent isolation
    test_valid_lora_and_concurrent_isolation();

    inferlet_deinit();
    std::cout << "========== All Step 15 Tests Passed Successfully! ==========" << std::endl;
    return 0;
}

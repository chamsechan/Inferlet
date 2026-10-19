#include "inferlet.h"
#include "support/object_registry.h"
#include "support/validation.h"
#include "support/status.h"
#include "support/last_error.h"
#include <cassert>
#include <iostream>
#include <vector>
#include <string>

using namespace inferlet;

void test_live_object_registry() {
    LiveObjectRegistry &reg = LiveObjectRegistry::instance();
    int dummy1 = 1;
    int dummy2 = 2;

    assert(!reg.is_registered(&dummy1));
    assert(!reg.is_valid(&dummy1, ObjectKind::Parameter));

    reg.register_object(&dummy1, ObjectKind::Parameter);
    assert(reg.is_registered(&dummy1));
    assert(reg.is_valid(&dummy1, ObjectKind::Parameter));
    assert(!reg.is_valid(&dummy1, ObjectKind::TaskInput));

    reg.unregister_object(&dummy1);
    assert(!reg.is_registered(&dummy1));
    assert(!reg.is_valid(&dummy1, ObjectKind::Parameter));

    // Double unregister should be safe
    reg.unregister_object(&dummy1);
    assert(!reg.is_registered(&dummy1));

    std::cout << "[PASS] test_live_object_registry\n";
}

void test_utf8_validation() {
    assert(is_valid_utf8("Hello, world!"));
    assert(is_valid_utf8("你好，世界！")); // 3-byte Chinese characters
    assert(is_valid_utf8("🌍🚀")); // 4-byte emojis
    assert(is_valid_utf8("")); // empty string

    // Invalid UTF-8 cases
    const char invalid1[] = {(char)0xFF, 0};
    assert(!is_valid_utf8(invalid1));

    const char invalid2[] = {(char)0xC0, (char)0xAF, 0}; // Overlong 2-byte
    assert(!is_valid_utf8(invalid2));

    const char invalid3[] = {(char)0xED, (char)0xA0, (char)0x80, 0}; // UTF-16 surrogate
    assert(!is_valid_utf8(invalid3));

    std::cout << "[PASS] test_utf8_validation\n";
}

void test_json_validation() {
    assert(is_valid_json("{\"name\": \"Inferlet\", \"version\": 1}"));
    assert(is_valid_json("[1, 2, 3]"));
    assert(!is_valid_json("not json"));
    assert(!is_valid_json("{broken json"));

    assert(is_valid_json_object("{\"key\": \"value\"}"));
    assert(!is_valid_json_object("[1, 2, 3]"));
    assert(!is_valid_json_object("\"just string\""));

    std::cout << "[PASS] test_json_validation\n";
}

void test_safe_multiply() {
    size_t out = 0;
    assert(safe_multiply<size_t>(10, 20, out) && out == 200);
    assert(safe_multiply<size_t>(0, 100, out) && out == 0);
    assert(safe_multiply<size_t>(100, 0, out) && out == 0);

    size_t max_val = std::numeric_limits<size_t>::max();
    assert(!safe_multiply<size_t>(max_val, 2, out));
    assert(!safe_multiply<size_t>(max_val / 2 + 1, 2, out));

    std::cout << "[PASS] test_safe_multiply\n";
}

void test_c_api_null_safety() {
    // Deallocations with NULL must not crash
    inferlet_parameter_deallocate(nullptr);
    inferlet_task_input_deallocate(nullptr);
    inferlet_task_output_deallocate(nullptr);
    inferlet_unload(nullptr);

    // Allocations and valid deallocations
    void *param = inferlet_parameter_allocate();
    assert(param != nullptr);
    inferlet_parameter_deallocate(param);

    void *in = inferlet_task_input_allocate();
    assert(in != nullptr);
    inferlet_task_input_deallocate(in);

    // Using deallocated handle should return invalid handle
    assert(inferlet_task_input_set_prompt_text(in, "hello") == INFERLET_E_INVALID_HANDLE);

    std::cout << "[PASS] test_c_api_null_safety\n";
}

void test_c_api_type_safety() {
    void *param = inferlet_parameter_allocate();
    assert(param != nullptr);

    // Pass Parameter handle to TaskInput API -> should fail with INVALID_HANDLE
    int ret = inferlet_task_input_set_prompt_text(param, "hello");
    assert(ret == INFERLET_E_INVALID_HANDLE);
    assert(get_last_error_code() == INFERLET_E_INVALID_HANDLE);

    // Pass wrong handle to Run
    void *out = nullptr;
    ret = inferlet_run(param, param, &out);
    assert(ret == INFERLET_E_INVALID_HANDLE);

    inferlet_parameter_deallocate(param);
    std::cout << "[PASS] test_c_api_type_safety\n";
}

void test_c_api_argument_validation() {
    void *in = inferlet_task_input_allocate();
    assert(in != nullptr);

    // NULL prompt text
    assert(inferlet_task_input_set_prompt_text(in, nullptr) == INFERLET_E_INVALID_ARGUMENT);

    // Invalid UTF-8 prompt
    const char bad_utf8[] = {(char)0xFF, (char)0xFE, 0};
    assert(inferlet_task_input_set_prompt_text(in, bad_utf8) == INFERLET_E_INVALID_ARGUMENT);

    // Range checks
    assert(inferlet_task_input_set_beam_size(in, 0) == INFERLET_E_INVALID_ARGUMENT);
    assert(inferlet_task_input_set_beam_size(in, -1) == INFERLET_E_INVALID_ARGUMENT);
    assert(inferlet_task_input_set_beam_size(in, 1) == INFERLET_OK);

    assert(inferlet_task_input_set_top_p(in, 0.0f) == INFERLET_E_INVALID_ARGUMENT);
    assert(inferlet_task_input_set_top_p(in, 1.5f) == INFERLET_E_INVALID_ARGUMENT);
    assert(inferlet_task_input_set_top_p(in, 0.95f) == INFERLET_OK);

    assert(inferlet_task_input_set_temperature(in, -0.1f) == INFERLET_E_INVALID_ARGUMENT);
    assert(inferlet_task_input_set_temperature(in, 0.0f) == INFERLET_OK);

    assert(inferlet_task_input_set_repetition_penalty(in, 0.0f) == INFERLET_E_INVALID_ARGUMENT);
    assert(inferlet_task_input_set_repetition_penalty(in, 1.1f) == INFERLET_OK);

    assert(inferlet_task_input_set_probability_type(in, -1) == INFERLET_E_INVALID_ARGUMENT);
    assert(inferlet_task_input_set_probability_type(in, 3) == INFERLET_E_INVALID_ARGUMENT);
    assert(inferlet_task_input_set_probability_type(in, 2) == INFERLET_OK);

    inferlet_task_input_deallocate(in);
    std::cout << "[PASS] test_c_api_argument_validation\n";
}

int main() {
    test_live_object_registry();
    test_utf8_validation();
    test_json_validation();
    test_safe_multiply();
    test_c_api_null_safety();
    test_c_api_type_safety();
    test_c_api_argument_validation();

    std::cout << "All validation and infrastructure tests passed successfully!\n";
    return 0;
}

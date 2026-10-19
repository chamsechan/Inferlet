#include "inferlet.h"
#include "core/runtime.h"
#include "support/last_error.h"
#include <cassert>
#include <iostream>
#include <cstring>
#include <cstdlib>

using namespace inferlet;

void test_uninitialized_load_fails() {
    // Ensure we start with clean uninitialized runtime
    assert(!Runtime::instance().is_initialized());

    void *handle = inferlet_load_from_file("dummy_model.gguf", nullptr);
    assert(handle == nullptr);
    assert(get_last_error_code() == INFERLET_E_INVALID_STATE);

    std::cout << "[PASS] test_uninitialized_load_fails\n";
}

void test_nested_init_deinit_loop() {
    const int iterations = 500;

    for (int i = 0; i < iterations; ++i) {
        inferlet_init();
        assert(Runtime::instance().is_initialized());
        assert(Runtime::instance().init_ref_count() == static_cast<uint32_t>(i + 1));
        assert(Runtime::instance().backend_initialized());
    }

    for (int i = iterations; i > 0; --i) {
        assert(Runtime::instance().init_ref_count() == static_cast<uint32_t>(i));
        inferlet_deinit();
    }

    assert(!Runtime::instance().is_initialized());
    assert(Runtime::instance().init_ref_count() == 0);
    assert(!Runtime::instance().backend_initialized());

    // Extra DeInit calls should be safe and no-ops
    inferlet_deinit();
    inferlet_deinit();
    assert(Runtime::instance().init_ref_count() == 0);
    assert(!Runtime::instance().backend_initialized());

    std::cout << "[PASS] test_nested_init_deinit_loop\n";
}

void test_deferred_shutdown() {
    inferlet_init();
    assert(Runtime::instance().is_initialized());
    assert(Runtime::instance().backend_initialized());

    const char *model_path = std::getenv("INFERLET_TEST_MODEL");
    if (!model_path || std::strlen(model_path) == 0) {
        model_path = "/home/ubuntu/project/llm-ops-agy/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    }

    void *param = inferlet_parameter_allocate();
    inferlet_parameter_set_tokenizer_only(param, 1);

    // Load a handle
    void *handle = inferlet_load_from_file(model_path, param);
    assert(handle != nullptr);
    assert(Runtime::instance().live_handle_count() == 1);
    inferlet_parameter_deallocate(param);

    // Call DeInit before Unload: deferred shutdown
    inferlet_deinit();
    assert(!Runtime::instance().is_initialized()); // No new Loads allowed
    assert(Runtime::instance().init_ref_count() == 0);
    assert(Runtime::instance().live_handle_count() == 1);
    // Backend must STILL be initialized until all handles are unloaded!
    assert(Runtime::instance().backend_initialized());

    // New Load should fail because init_ref_count == 0
    void *handle2 = inferlet_load_from_file("test_model2.gguf", nullptr);
    assert(handle2 == nullptr);
    assert(get_last_error_code() == INFERLET_E_INVALID_STATE);

    // Now unload the handle
    inferlet_unload(handle);
    assert(Runtime::instance().live_handle_count() == 0);
    // Now backend should be safely freed
    assert(!Runtime::instance().backend_initialized());

    // Re-initialization should work cleanly
    inferlet_init();
    assert(Runtime::instance().is_initialized());
    assert(Runtime::instance().backend_initialized());
    inferlet_deinit();
    assert(!Runtime::instance().is_initialized());
    assert(!Runtime::instance().backend_initialized());

    std::cout << "[PASS] test_deferred_shutdown\n";
}

int main() {
    test_uninitialized_load_fails();
    test_nested_init_deinit_loop();
    test_deferred_shutdown();

    std::cout << "All Runtime lifecycle tests passed successfully!\n";
    return 0;
}

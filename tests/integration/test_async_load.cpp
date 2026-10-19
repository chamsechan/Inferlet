#include "inferlet.h"
#include "support/last_error.h"
#include <cassert>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <cstdlib>
#include <cstring>

static const char *get_model_path() {
    const char *env = std::getenv("INFERLET_TEST_MODEL");
    if (env && std::strlen(env) > 0) {
        return env;
    }
    return "/home/ubuntu/project/llm-ops-agy/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
}

static void test_async_load_success_and_run() {
    std::cout << "Running test_async_load_success_and_run...\n";
    void *param = inferlet_parameter_allocate();
    inferlet_parameter_set_load_from_file_sync(param, 0); // async mode

    const char *model_path = get_model_path();
    void *handle = inferlet_load_from_file(model_path, param);
    inferlet_parameter_deallocate(param);
    assert(handle != nullptr);

    // Initial check: should be active (either loading or ready)
    int active = 0;
    int ret = inferlet_check_backend_active(handle, &active);
    assert(ret == INFERLET_OK);

    // Sync Run waits for model to finish loading if needed
    void *task_in = inferlet_task_input_allocate();
    inferlet_task_input_set_prompt_text(task_in, "Hello world");
    inferlet_task_input_set_max_output_tokens(task_in, 5);

    void *task_out = nullptr;
    ret = inferlet_run(handle, task_in, &task_out);
    assert(ret == INFERLET_OK);
    assert(task_out != nullptr);

    const char *text = inferlet_task_output_get_result_text(task_out);
    assert(text != nullptr && std::strlen(text) > 0);

    inferlet_task_input_deallocate(task_in);
    inferlet_task_output_deallocate(task_out);
    inferlet_unload(handle);
    std::cout << "[PASS] test_async_load_success_and_run\n";
}

static void test_sync_load_success_and_run() {
    std::cout << "Running test_sync_load_success_and_run...\n";
    void *param = inferlet_parameter_allocate();
    inferlet_parameter_set_load_from_file_sync(param, 1); // sync mode

    const char *model_path = get_model_path();
    void *handle = inferlet_load_from_file(model_path, param);
    inferlet_parameter_deallocate(param);
    assert(handle != nullptr);

    void *task_in = inferlet_task_input_allocate();
    inferlet_task_input_set_prompt_text(task_in, "Once upon a time");
    inferlet_task_input_set_max_output_tokens(task_in, 5);

    void *task_out = nullptr;
    int ret = inferlet_run(handle, task_in, &task_out);
    assert(ret == INFERLET_OK);
    assert(task_out != nullptr);

    inferlet_task_input_deallocate(task_in);
    inferlet_task_output_deallocate(task_out);
    inferlet_unload(handle);
    std::cout << "[PASS] test_sync_load_success_and_run\n";
}

static void test_shared_model_concurrent_load() {
    std::cout << "Running test_shared_model_concurrent_load...\n";
    const char *model_path = get_model_path();

    void *param1 = inferlet_parameter_allocate();
    inferlet_parameter_set_load_from_file_sync(param1, 0); // async

    void *param2 = inferlet_parameter_allocate();
    inferlet_parameter_set_load_from_file_sync(param2, 1); // sync

    void *handle1 = nullptr;
    void *handle2 = nullptr;

    std::thread t1([&]() {
        handle1 = inferlet_load_from_file(model_path, param1);
    });
    std::thread t2([&]() {
        handle2 = inferlet_load_from_file(model_path, param2);
    });

    t1.join();
    t2.join();

    inferlet_parameter_deallocate(param1);
    inferlet_parameter_deallocate(param2);

    assert(handle1 != nullptr);
    assert(handle2 != nullptr);

    // Test different fingerprint on already loaded model -> must fail immediately
    void *param_diff = inferlet_parameter_allocate();
    inferlet_parameter_set_tokenizer_only(param_diff, 1); // differs from full model
    void *handle_diff = inferlet_load_from_file(model_path, param_diff);
    assert(handle_diff == nullptr);
    assert(inferlet::get_last_error_code() == INFERLET_E_INVALID_ARGUMENT);
    inferlet_parameter_deallocate(param_diff);

    inferlet_unload(handle1);
    inferlet_unload(handle2);
    std::cout << "[PASS] test_shared_model_concurrent_load\n";
}

static void test_async_load_nonexistent_model() {
    std::cout << "Running test_async_load_nonexistent_model...\n";
    void *param = inferlet_parameter_allocate();
    inferlet_parameter_set_load_from_file_sync(param, 0); // async

    void *handle = inferlet_load_from_file("/tmp/inferlet_nonexistent_model_xyz.gguf", param);
    inferlet_parameter_deallocate(param);
    assert(handle != nullptr);

    // Poll CheckBackendActive until failure is observed
    int ret = 0;
    int active = 0;
    for (int i = 0; i < 50; ++i) {
        ret = inferlet_check_backend_active(handle, &active);
        if (ret != INFERLET_OK) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    assert(ret == INFERLET_E_MODEL_LOAD);
    assert(inferlet::get_last_error_code() == INFERLET_E_MODEL_LOAD);

    // Test synchronous Run -> must return INFERLET_E_MODEL_LOAD
    void *task_in = inferlet_task_input_allocate();
    inferlet_task_input_set_prompt_text(task_in, "Test prompt");
    void *task_out = nullptr;
    ret = inferlet_run(handle, task_in, &task_out);
    assert(ret == INFERLET_E_MODEL_LOAD);
    assert(task_out == nullptr);

    // Test async Run -> task fails with INFERLET_E_MODEL_LOAD
    void *task = nullptr;
    ret = inferlet_run_async(handle, task_in, &task);
    assert(ret == INFERLET_OK);
    assert(task != nullptr);

    void *async_out = nullptr;
    ret = inferlet_wait_task(handle, task, &async_out);
    assert(ret == INFERLET_E_MODEL_LOAD);
    assert(async_out == nullptr);

    inferlet_task_input_deallocate(task_in);

    // Unload must not deadlock
    inferlet_unload(handle);
    std::cout << "[PASS] test_async_load_nonexistent_model\n";
}

static void test_sync_load_nonexistent_model() {
    std::cout << "Running test_sync_load_nonexistent_model...\n";
    void *param = inferlet_parameter_allocate();
    inferlet_parameter_set_load_from_file_sync(param, 1); // sync

    void *handle = inferlet_load_from_file("/tmp/inferlet_nonexistent_model_xyz.gguf", param);
    assert(handle == nullptr);
    assert(inferlet::get_last_error_code() == INFERLET_E_MODEL_LOAD);
    inferlet_parameter_deallocate(param);
    std::cout << "[PASS] test_sync_load_nonexistent_model\n";
}

static void test_corrupt_model() {
    std::cout << "Running test_corrupt_model...\n";
    const std::string corrupt_path = "/tmp/inferlet_test_corrupt.gguf";
    {
        std::ofstream ofs(corrupt_path, std::ios::binary);
        ofs << "THIS_IS_DEFINITELY_NOT_A_VALID_GGUF_FILE_HEADER_1234567890";
    }

    // 1. Sync load -> returns NULL immediately
    void *param_sync = inferlet_parameter_allocate();
    inferlet_parameter_set_load_from_file_sync(param_sync, 1);
    void *handle_sync = inferlet_load_from_file(corrupt_path.c_str(), param_sync);
    assert(handle_sync == nullptr);
    assert(inferlet::get_last_error_code() == INFERLET_E_MODEL_LOAD);
    inferlet_parameter_deallocate(param_sync);

    // 2. Async load -> returns handle, then Check/Run observe MODEL_LOAD
    void *param_async = inferlet_parameter_allocate();
    inferlet_parameter_set_load_from_file_sync(param_async, 0);
    void *handle_async = inferlet_load_from_file(corrupt_path.c_str(), param_async);
    inferlet_parameter_deallocate(param_async);
    assert(handle_async != nullptr);

    int ret = 0;
    int active = 0;
    for (int i = 0; i < 50; ++i) {
        ret = inferlet_check_backend_active(handle_async, &active);
        if (ret != INFERLET_OK) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    assert(ret == INFERLET_E_MODEL_LOAD);

    void *task_in = inferlet_task_input_allocate();
    inferlet_task_input_set_prompt_text(task_in, "Test");
    void *task_out = nullptr;
    ret = inferlet_run(handle_async, task_in, &task_out);
    assert(ret == INFERLET_E_MODEL_LOAD);
    assert(task_out == nullptr);

    inferlet_task_input_deallocate(task_in);
    inferlet_unload(handle_async);

    std::remove(corrupt_path.c_str());
    std::cout << "[PASS] test_corrupt_model\n";
}

static void test_unload_while_loading() {
    std::cout << "Running test_unload_while_loading...\n";
    const char *model_path = get_model_path();
    void *param = inferlet_parameter_allocate();
    inferlet_parameter_set_load_from_file_sync(param, 0); // async

    void *handle = inferlet_load_from_file(model_path, param);
    inferlet_parameter_deallocate(param);
    assert(handle != nullptr);

    // Immediately unload without waiting
    inferlet_unload(handle);
    std::cout << "[PASS] test_unload_while_loading\n";
}

static void test_cancel_task_waiting_for_model() {
    std::cout << "Running test_cancel_task_waiting_for_model...\n";
    const char *model_path = get_model_path();
    void *param = inferlet_parameter_allocate();
    inferlet_parameter_set_load_from_file_sync(param, 0); // async

    void *handle = inferlet_load_from_file(model_path, param);
    inferlet_parameter_deallocate(param);
    assert(handle != nullptr);

    void *task_in = inferlet_task_input_allocate();
    inferlet_task_input_set_prompt_text(task_in, "First prompt to cancel");
    inferlet_task_input_set_max_output_tokens(task_in, 10);

    void *task1 = nullptr;
    int ret = inferlet_run_async(handle, task_in, &task1);
    assert(ret == INFERLET_OK);

    // Immediately cancel the task
    ret = inferlet_cancel_task(handle, task1);
    assert(ret == INFERLET_OK);

    void *task_out1 = nullptr;
    ret = inferlet_wait_task(handle, task1, &task_out1);
    assert(ret == INFERLET_E_CANCELLED);
    assert(task_out1 == nullptr);

    // The shared model loading continues and succeeds; second task must complete
    inferlet_task_input_set_prompt_text(task_in, "Second prompt that must succeed");
    void *task2 = nullptr;
    ret = inferlet_run_async(handle, task_in, &task2);
    assert(ret == INFERLET_OK);

    void *task_out2 = nullptr;
    ret = inferlet_wait_task(handle, task2, &task_out2);
    assert(ret == INFERLET_OK);
    assert(task_out2 != nullptr);

    const char *text = inferlet_task_output_get_result_text(task_out2);
    assert(text != nullptr && std::strlen(text) > 0);

    inferlet_task_input_deallocate(task_in);
    inferlet_task_output_deallocate(task_out2);
    inferlet_unload(handle);
    std::cout << "[PASS] test_cancel_task_waiting_for_model\n";
}

int main() {
    std::cout << "================ Test Async Model Loading ================\n";
    inferlet_init();

    test_sync_load_nonexistent_model();
    test_async_load_nonexistent_model();
    test_corrupt_model();
    test_unload_while_loading();
    test_sync_load_success_and_run();
    test_async_load_success_and_run();
    test_shared_model_concurrent_load();
    test_cancel_task_waiting_for_model();

    inferlet_deinit();
    std::cout << "================ All Async Load Tests Passed! ================\n";
    return 0;
}

#include "inferlet.h"
#include <cassert>
#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <chrono>
#include <atomic>
#include "core/inference_task.h"

static const char *get_model_path() {
    const char *env = std::getenv("INFERLET_TEST_MODEL");
    if (env && std::strlen(env) > 0) {
        return env;
    }
    return "/home/ubuntu/project/llm-ops-agy/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
}

static void test_async_basic_wait(void *handle) {
    std::cout << "Running test_async_basic_wait...\n";
    void *task_in = inferlet_task_input_allocate();
    assert(task_in != nullptr);

    inferlet_task_input_set_prompt_text(task_in, "The color of the sky is usually");
    inferlet_task_input_set_max_output_tokens(task_in, 6);

    void *task = nullptr;
    int ret = inferlet_run_async(handle, task_in, &task);
    assert(ret == INFERLET_OK);
    assert(task != nullptr);

    int active = 0;
    ret = inferlet_check_backend_active(handle, &active);
    assert(ret == INFERLET_OK);

    // A completed task may still own an unconsumed result, but the backend
    // must already be idle. Wait internally without consuming the ticket.
    auto *ticket = inferlet::checked_cast<inferlet::TaskTicketInternal>(
        task, inferlet::ObjectKind::TaskTicket);
    assert(ticket != nullptr);
    {
        std::unique_lock<std::mutex> lock(ticket->core->mutex);
        ticket->core->cv.wait(lock, [&] { return ticket->core->is_terminal(); });
    }
    assert(inferlet_check_backend_active(handle, &active) == INFERLET_OK);
    assert(active == 0);

    void *task_out = nullptr;
    ret = inferlet_wait_task(handle, task, &task_out);
    assert(ret == INFERLET_OK);
    assert(task_out != nullptr);

    const char *text = inferlet_task_output_get_result_text(task_out);
    assert(text != nullptr);
    assert(std::strlen(text) > 0);
    std::cout << "Async output: \"" << text << "\"\n";

    // Double consume test: ticket was destroyed, using it must return invalid handle
    void *task_out2 = nullptr;
    ret = inferlet_wait_task(handle, task, &task_out2);
    assert(ret == INFERLET_E_INVALID_HANDLE);
    assert(task_out2 == nullptr);

    inferlet_task_input_deallocate(task_in);
    inferlet_task_output_deallocate(task_out);
    std::cout << "[PASS] test_async_basic_wait\n";
}

static void test_async_query_polling(void *handle) {
    std::cout << "Running test_async_query_polling...\n";
    void *task_in = inferlet_task_input_allocate();
    inferlet_task_input_set_prompt_text(task_in, "A sunny day is");
    inferlet_task_input_set_max_output_tokens(task_in, 6);

    void *task = nullptr;
    int ret = inferlet_run_async(handle, task_in, &task);
    assert(ret == INFERLET_OK);
    assert(task != nullptr);

    void *task_out = nullptr;
    while (true) {
        ret = inferlet_query_task(handle, task, &task_out);
        assert(ret == INFERLET_OK);
        if (task_out != nullptr) {
            break; // Finished
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    const char *text = inferlet_task_output_get_result_text(task_out);
    assert(text != nullptr && std::strlen(text) > 0);

    // Double consume with Query
    void *task_out2 = nullptr;
    ret = inferlet_query_task(handle, task, &task_out2);
    assert(ret == INFERLET_E_INVALID_HANDLE);

    inferlet_task_input_deallocate(task_in);
    inferlet_task_output_deallocate(task_out);
    std::cout << "[PASS] test_async_query_polling\n";
}

static void test_cancel_queued_task(void *handle) {
    std::cout << "Running test_cancel_queued_task...\n";
    void *task_in = inferlet_task_input_allocate();
    inferlet_task_input_set_prompt_text(task_in, "Tell me a long story about dragons");
    inferlet_task_input_set_max_output_tokens(task_in, 50);

    void *task = nullptr;
    int ret = inferlet_run_async(handle, task_in, &task);
    assert(ret == INFERLET_OK);

    // Immediately cancel
    ret = inferlet_cancel_task(handle, task);
    assert(ret == INFERLET_OK);

    // Repeat cancel should return 0
    ret = inferlet_cancel_task(handle, task);
    assert(ret == INFERLET_OK);

    void *task_out = nullptr;
    ret = inferlet_wait_task(handle, task, &task_out);
    assert(ret == INFERLET_E_CANCELLED);
    assert(task_out == nullptr);

    inferlet_task_input_deallocate(task_in);
    std::cout << "[PASS] test_cancel_queued_task\n";
}

static void test_cancel_running_task(void *handle) {
    std::cout << "Running test_cancel_running_task...\n";
    void *task_in = inferlet_task_input_allocate();
    inferlet_task_input_set_prompt_text(task_in, "Count numbers: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,");
    inferlet_task_input_set_max_output_tokens(task_in, 100);

    void *task = nullptr;
    int ret = inferlet_run_async(handle, task_in, &task);
    assert(ret == INFERLET_OK);

    // Sleep briefly so worker starts execution
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    ret = inferlet_cancel_task(handle, task);
    assert(ret == INFERLET_OK);

    void *task_out = nullptr;
    ret = inferlet_wait_task(handle, task, &task_out);
    assert(ret == INFERLET_E_CANCELLED);
    assert(task_out == nullptr);

    inferlet_task_input_deallocate(task_in);
    std::cout << "[PASS] test_cancel_running_task\n";
}

static void test_query_wait_race(void *handle) {
    std::cout << "Running test_query_wait_race...\n";
    void *task_in = inferlet_task_input_allocate();
    inferlet_task_input_set_prompt_text(task_in, "Hello!");
    inferlet_task_input_set_max_output_tokens(task_in, 5);

    void *task = nullptr;
    int ret = inferlet_run_async(handle, task_in, &task);
    assert(ret == INFERLET_OK);

    void *out_wait = nullptr;
    void *out_query = nullptr;
    int ret_wait = -999;
    int ret_query = -999;

    std::thread t_wait([&]() {
        ret_wait = inferlet_wait_task(handle, task, &out_wait);
    });

    std::thread t_query([&]() {
        while (true) {
            ret_query = inferlet_query_task(handle, task, &out_query);
            if (ret_query != INFERLET_OK || out_query != nullptr) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    });

    t_wait.join();
    t_query.join();

    // Exactly one must have gotten the output, the other got invalid handle
    int success_count = 0;
    if (ret_wait == INFERLET_OK && out_wait != nullptr) {
        success_count++;
        inferlet_task_output_deallocate(out_wait);
    }
    if (ret_query == INFERLET_OK && out_query != nullptr) {
        success_count++;
        inferlet_task_output_deallocate(out_query);
    }
    assert(success_count == 1);

    inferlet_task_input_deallocate(task_in);
    std::cout << "[PASS] test_query_wait_race\n";
}

static void test_unload_race_with_in_flight_tasks() {
    std::cout << "Running test_unload_race_with_in_flight_tasks...\n";
    const char *model_path = get_model_path();
    void *param = inferlet_parameter_allocate();
    void *handle = inferlet_load_from_file(model_path, param);
    inferlet_parameter_deallocate(param);
    assert(handle != nullptr);

    void *task_in = inferlet_task_input_allocate();
    inferlet_task_input_set_prompt_text(task_in, "Generate words:");
    inferlet_task_input_set_max_output_tokens(task_in, 50);

    std::vector<void *> tickets;
    for (int i = 0; i < 4; ++i) {
        void *t = nullptr;
        int ret = inferlet_run_async(handle, task_in, &t);
        assert(ret == INFERLET_OK);
        tickets.push_back(t);
    }

    // Unload must cancel and drain without deadlock
    inferlet_unload(handle);

    inferlet_task_input_deallocate(task_in);
    std::cout << "[PASS] test_unload_race_with_in_flight_tasks\n";
}

int main() {
    std::cout << "================ Test Async Inference ================\n";
    inferlet_init();

    const char *model_path = get_model_path();
    std::cout << "Loading model: " << model_path << "\n";

    void *param = inferlet_parameter_allocate();
    void *handle = inferlet_load_from_file(model_path, param);
    inferlet_parameter_deallocate(param);

    if (!handle) {
        std::cerr << "Failed to load model from " << model_path << "\n";
        inferlet_deinit();
        return 1;
    }

    test_async_basic_wait(handle);
    test_async_query_polling(handle);
    test_cancel_queued_task(handle);
    test_cancel_running_task(handle);
    test_query_wait_race(handle);

    inferlet_unload(handle);

    // Test unload race with tasks in-flight
    test_unload_race_with_in_flight_tasks();

    inferlet_deinit();
    std::cout << "================ All Async Tests Passed! ================\n";
    return 0;
}

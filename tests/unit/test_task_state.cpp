#include "core/inference_task.h"
#include "core/model_registry.h"
#include <cassert>
#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>

using namespace inferlet;

static void test_task_state_transitions() {
    std::cout << "Running test_task_state_transitions...\n";
    TaskCore core;
    assert(core.state == TaskState::Queued);
    assert(!core.is_terminal());

    core.state = TaskState::Running;
    assert(!core.is_terminal());

    // First terminal transition succeeds
    bool ok = core.set_terminal_state(TaskState::Completed, Status::ok());
    assert(ok);
    assert(core.is_terminal());
    assert(core.state == TaskState::Completed);

    // Terminal state cannot be overwritten!
    ok = core.set_terminal_state(TaskState::Cancelled, Status::cancelled());
    assert(!ok);
    assert(core.state == TaskState::Completed);

    ok = core.set_terminal_state(TaskState::Failed, Status::inference_error());
    assert(!ok);
    assert(core.state == TaskState::Completed);

    std::cout << "[PASS] test_task_state_transitions\n";
}

static void test_task_state_concurrent_race() {
    std::cout << "Running test_task_state_concurrent_race...\n";
    TaskCore core;
    core.state = TaskState::Running;

    std::atomic<int> winners{0};
    const int num_threads = 16;
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&core, &winners, i]() {
            TaskState target = (i % 2 == 0) ? TaskState::Completed : TaskState::Cancelled;
            Status st = (i % 2 == 0) ? Status::ok() : Status::cancelled();
            if (core.set_terminal_state(target, st)) {
                winners++;
            }
        });
    }

    for (auto &t : threads) {
        t.join();
    }

    // Exactly one thread won the race
    assert(winners.load() == 1);
    assert(core.is_terminal());

    std::cout << "[PASS] test_task_state_concurrent_race\n";
}

static void test_semaphore() {
    std::cout << "Running test_semaphore...\n";
    Semaphore sem(2);
    assert(sem.count() == 2);

    assert(sem.try_acquire());
    assert(sem.count() == 1);

    assert(sem.try_acquire());
    assert(sem.count() == 0);

    // Next try_acquire should fail
    assert(!sem.try_acquire());

    // try_acquire_for with timeout should fail
    auto start = std::chrono::steady_clock::now();
    bool acquired = sem.try_acquire_for(std::chrono::milliseconds(20));
    auto elapsed = std::chrono::steady_clock::now() - start;
    assert(!acquired);
    assert(elapsed >= std::chrono::milliseconds(15));

    // Release one slot
    sem.release();
    assert(sem.count() == 1);

    // Now try_acquire_for succeeds
    assert(sem.try_acquire_for(std::chrono::milliseconds(20)));
    assert(sem.count() == 0);

    // Release both
    sem.release();
    sem.release();
    assert(sem.count() == 2);

    std::cout << "[PASS] test_semaphore\n";
}

int main() {
    std::cout << "================ Test Task State ================\n";
    test_task_state_transitions();
    test_task_state_concurrent_race();
    test_semaphore();
    std::cout << "================ All Task State Tests Passed! ================\n";
    return 0;
}

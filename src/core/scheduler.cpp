#include "scheduler.h"
#include "inference_task.h"
#include "stream_sink.h"
#include "engine/generator.h"
#include "engine/result_sink.h"
#include "support/logging.h"
#include <chrono>

namespace inferlet {

Scheduler::~Scheduler() {
    stop();
}

void Scheduler::start(size_t worker_count) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (running_) return;

    if (worker_count == 0) {
        unsigned int hw = std::thread::hardware_concurrency();
        worker_count = (hw > 0) ? std::max(2u, hw) : 4;
    }

    running_ = true;
    workers_.reserve(worker_count);
    for (size_t i = 0; i < worker_count; ++i) {
        workers_.emplace_back(&Scheduler::worker_loop, this);
    }
    log_debug("Scheduler started with " + std::to_string(worker_count) + " workers");
}

void Scheduler::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!running_) return;
        running_ = false;
        cv_.notify_all();
    }

    for (auto &w : workers_) {
        if (w.joinable()) {
            w.join();
        }
    }
    workers_.clear();

    std::lock_guard<std::mutex> lock(mutex_);
    while (!queue_.empty()) {
        auto task = queue_.front();
        queue_.pop_front();
        if (task && task->model) {
            task->model->queued--;
            task->set_terminal_state(TaskState::Cancelled, Status::cancelled("Scheduler stopped"));
        }
    }
}

bool Scheduler::is_running() const {
    return running_.load();
}

Status Scheduler::submit(std::shared_ptr<TaskCore> task) {
    if (!task || !task->model) {
        return Status::invalid_argument("Cannot submit null task or null model");
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (!running_) {
        return Status::invalid_state("Scheduler is not running");
    }

    const int queue_capacity = task->model->config.parsed_run_config.scheduler.queue_capacity;
    if (queue_capacity > 0 && task->model->queued.load() >= queue_capacity) {
        return Status::invalid_state("Model task queue capacity exceeded");
    }

    if (task->cancel_requested.load()) {
        task->set_terminal_state(TaskState::Cancelled, Status::cancelled());
        return Status::ok();
    }

    {
        std::lock_guard<std::mutex> tlock(task->mutex);
        task->state = TaskState::Queued;
    }
    task->model->queued++;
    queue_.push_back(task);
    cv_.notify_one();
    return Status::ok();
}

void Scheduler::worker_loop() {
    while (running_) {
        std::shared_ptr<TaskCore> task;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this]() {
                return !running_ || !queue_.empty();
            });

            if (!running_ && queue_.empty()) {
                break;
            }

            if (!queue_.empty()) {
                task = queue_.front();
                queue_.pop_front();
            }
        }

        if (!task) continue;

        auto model = task->model;

        // 1. Check if cancelled while in queue
        if (task->cancel_requested.load()) {
            if (model) model->queued--;
            task->set_terminal_state(TaskState::Cancelled, Status::cancelled());
            continue;
        }

        // 2. Transition to WaitingForModel
        bool cancelled_before_wait = false;
        {
            std::lock_guard<std::mutex> tlock(task->mutex);
            if (task->cancel_requested.load() || task->is_terminal()) {
                cancelled_before_wait = true;
            } else {
                task->state = TaskState::WaitingForModel;
                task->cv.notify_all();
            }
        }
        if (cancelled_before_wait) {
            if (model) model->queued--;
            task->set_terminal_state(TaskState::Cancelled, Status::cancelled());
            continue;
        }

        // Wait for model load if still loading
        LoadState model_load_state = LoadState::Ready;
        Status model_load_status;
        if (model) {
            std::unique_lock<std::mutex> mlock(model->mutex);
            while (model->load_state == LoadState::Loading && !task->cancel_requested.load() && running_) {
                model->load_cv.wait_for(mlock, std::chrono::milliseconds(50));
            }
            model_load_state = model->load_state;
            model_load_status = model->load_status;
        }

        if (task->cancel_requested.load() || !running_) {
            if (model) model->queued--;
            task->set_terminal_state(TaskState::Cancelled, Status::cancelled());
            continue;
        }

        if (model && model_load_state == LoadState::Failed) {
            if (model) model->queued--;
            task->set_terminal_state(TaskState::Failed,
                                     model_load_status.is_ok() ? Status::model_load() : model_load_status);
            continue;
        }

        // 3. Acquire model execution slot (semaphore)
        bool slot_acquired = false;
        if (model && model->exec_semaphore) {
            while (running_ && !task->cancel_requested.load()) {
                if (model->exec_semaphore->try_acquire_for(std::chrono::milliseconds(20))) {
                    slot_acquired = true;
                    break;
                }
            }
        } else {
            slot_acquired = true;
        }

        if (!slot_acquired || task->cancel_requested.load() || !running_) {
            if (slot_acquired && model && model->exec_semaphore) {
                model->exec_semaphore->release();
            }
            if (model) model->queued--;
            task->set_terminal_state(TaskState::Cancelled, Status::cancelled());
            continue;
        }

        // 4. Transition to Running
        if (model) {
            model->queued--;
            model->running++;
        }

        bool cancelled_before_run = false;
        {
            std::lock_guard<std::mutex> tlock(task->mutex);
            if (task->cancel_requested.load() || task->is_terminal()) {
                cancelled_before_run = true;
            } else {
                task->state = TaskState::Running;
                task->cv.notify_all();
            }
        }
        if (cancelled_before_run) {
            if (model && model->exec_semaphore) {
                model->exec_semaphore->release();
            }
            if (model) model->running--;
            task->set_terminal_state(TaskState::Cancelled, Status::cancelled());
            continue;
        }

        // 5. Execute inference without holding management locks
        if (task->streaming) {
            StreamingSink sink(task, model ? model->config.parsed_run_config.limits.max_verbose_bytes : 0);
            Status st = Generator::run(task->request, model, sink, &task->cancel_requested);
            if (st.is_ok() && !sink.finish_status().is_ok()) {
                st = sink.finish_status();
            }

            // 6. Release model execution slot
            if (model && model->exec_semaphore) {
                model->exec_semaphore->release();
            }
            if (model) model->running--;

            if (task->cancel_requested.load()) {
                task->set_terminal_state(TaskState::Cancelled, Status::cancelled());
            } else if (!st) {
                task->set_terminal_state(TaskState::Failed, st);
            } else {
                task->set_terminal_state(TaskState::Completed, Status::ok());
            }
        } else {
            auto out = std::make_unique<TaskOutputInternal>();
            CollectingSink sink(*out, model ? model->config.parsed_run_config.limits.max_verbose_bytes : 0);
            Status st = Generator::run(task->request, model, sink, &task->cancel_requested);
            if (st.is_ok() && !sink.finish_status().is_ok()) {
                st = sink.finish_status();
            }

            // 6. Release model execution slot
            if (model && model->exec_semaphore) {
                model->exec_semaphore->release();
            }
            if (model) model->running--;

            if (task->cancel_requested.load()) {
                task->set_terminal_state(TaskState::Cancelled, Status::cancelled());
            } else if (!st) {
                task->set_terminal_state(TaskState::Failed, st);
            } else {
                task->output = std::move(out);
                task->set_terminal_state(TaskState::Completed, Status::ok());
            }
        }
    }
}

} // namespace inferlet

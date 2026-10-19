#ifndef INFERLET_CORE_INFERENCE_TASK_H_INCLUDED
#define INFERLET_CORE_INFERENCE_TASK_H_INCLUDED

#include "support/object_registry.h"
#include "support/status.h"
#include "core/task_input.h"
#include "core/task_output.h"
#include "core/model_registry.h"
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <memory>
#include <deque>
#include <optional>
#include <string>

namespace inferlet {

struct ModelHandleInternal;

enum class TaskState {
    Queued,
    WaitingForModel,
    Running,
    Cancelling,
    Completed,
    Failed,
    Cancelled,
    Consumed
};

const char *task_state_name(TaskState s);

struct StreamChunk {
    TokenResult result;
    std::optional<InputTokenInfo> input;
};

struct TaskCore {
    mutable std::mutex mutex;
    std::condition_variable cv;
    TaskState state = TaskState::Queued;
    std::atomic<bool> cancel_requested{false};
    Status terminal_status;
    TaskRequest request;
    std::shared_ptr<SharedModel> model;
    std::unique_ptr<TaskOutputInternal> output;

    bool streaming = false;
    std::deque<StreamChunk> chunks;
    std::optional<StreamChunk> current_chunk;
    size_t max_queued_chunks = 16;
    std::condition_variable queue_not_full;

    bool is_terminal() const {
        return state == TaskState::Completed || state == TaskState::Failed ||
               state == TaskState::Cancelled || state == TaskState::Consumed;
    }

    bool set_terminal_state(TaskState new_state, Status status) {
        std::lock_guard<std::mutex> lock(mutex);
        if (is_terminal()) {
            return false;
        }
        state = new_state;
        terminal_status = status;
        model.reset();
        cv.notify_all();
        queue_not_full.notify_all();
        return true;
    }
};

struct TaskTicketInternal {
    ObjectHeader header{INFERLET_OBJECT_MAGIC, ObjectKind::TaskTicket, INFERLET_ABI_VERSION};
    std::shared_ptr<TaskCore> core;
    ModelHandleInternal *owner = nullptr;
    bool consuming = false;
};

struct StreamTicketInternal {
    ObjectHeader header{INFERLET_OBJECT_MAGIC, ObjectKind::StreamTicket, INFERLET_ABI_VERSION};
    std::shared_ptr<TaskCore> core;
    ModelHandleInternal *owner = nullptr;
};

} // namespace inferlet

#endif // INFERLET_CORE_INFERENCE_TASK_H_INCLUDED

#include "inference_task.h"

namespace inferlet {

const char *task_state_name(TaskState s) {
    switch (s) {
        case TaskState::Queued: return "Queued";
        case TaskState::WaitingForModel: return "WaitingForModel";
        case TaskState::Running: return "Running";
        case TaskState::Cancelling: return "Cancelling";
        case TaskState::Completed: return "Completed";
        case TaskState::Failed: return "Failed";
        case TaskState::Cancelled: return "Cancelled";
        case TaskState::Consumed: return "Consumed";
        default: return "Unknown";
    }
}

} // namespace inferlet

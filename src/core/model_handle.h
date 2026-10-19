#ifndef INFERLET_CORE_MODEL_HANDLE_H_INCLUDED
#define INFERLET_CORE_MODEL_HANDLE_H_INCLUDED

#include "support/object_registry.h"
#include "model_registry.h"
#include <thread>
#include <memory>
#include <mutex>
#include <unordered_set>

namespace inferlet {

struct TaskCore;
struct TaskTicketInternal;
struct StreamTicketInternal;

struct ModelHandleInternal {
    ObjectHeader header{INFERLET_OBJECT_MAGIC, ObjectKind::ModelHandle, INFERLET_ABI_VERSION};
    std::shared_ptr<SharedModel> shared_model;
    std::thread::id owner_thread;
    std::mutex task_mutex;
    bool closing = false;
    std::unordered_set<std::shared_ptr<TaskCore>> tasks;
    std::unordered_set<TaskTicketInternal *> task_tickets;
    std::unordered_set<StreamTicketInternal *> stream_tickets;

    ModelHandleInternal(std::shared_ptr<SharedModel> model)
        : shared_model(std::move(model)), owner_thread(std::this_thread::get_id()) {}

    Status wait_for_ready() const {
        if (!shared_model) return Status::invalid_handle("ModelHandle has no shared model");
        {
            std::unique_lock<std::mutex> lock(shared_model->mutex);
            if (shared_model->load_state == LoadState::Loading) {
                shared_model->load_cv.wait(lock, [this]() {
                    return shared_model->load_state != LoadState::Loading;
                });
            }
            if (shared_model->load_state == LoadState::Failed) {
                return shared_model->load_status.is_ok() ? Status::model_load() : shared_model->load_status;
            }
        }
        return Status::ok();
    }

    const llama_vocab *vocab() const {
        return shared_model ? shared_model->vocab : nullptr;
    }

    bool is_tokenizer_only() const {
        return shared_model ? shared_model->config.tokenizer_only : false;
    }
};

} // namespace inferlet

#endif // INFERLET_CORE_MODEL_HANDLE_H_INCLUDED

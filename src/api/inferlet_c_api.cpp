#include "inferlet.h"
#include "support/status.h"
#include "support/last_error.h"
#include "support/logging.h"
#include "support/object_registry.h"
#include "support/validation.h"
#include "support/internal_types.h"

#include <new>
#include <exception>
#include <cstring>
#include <limits>
#include "core/runtime.h"
#include "core/model_registry.h"
#include "engine/token_codec.h"
#include "engine/generator.h"
#include "engine/result_sink.h"

using namespace inferlet;

namespace {

// Synchronous calls share the same execution limit and activity counters as
// scheduled calls, including when different handles refer to the same model.
class SyncExecutionGuard {
public:
    explicit SyncExecutionGuard(std::shared_ptr<SharedModel> model) : model_(std::move(model)) {
        ++model_->queued;
        try {
            model_->exec_semaphore->acquire();
        } catch (...) {
            --model_->queued;
            throw;
        }
        ++model_->running;
        --model_->queued;
    }
    ~SyncExecutionGuard() {
        --model_->running;
        model_->exec_semaphore->release();
    }
    SyncExecutionGuard(const SyncExecutionGuard &) = delete;
    SyncExecutionGuard &operator=(const SyncExecutionGuard &) = delete;
private:
    std::shared_ptr<SharedModel> model_;
};

LogLevel configured_log_level(const std::string &level) {
    if (level == "debug") return LogLevel::Debug;
    if (level == "warn") return LogLevel::Warn;
    if (level == "error") return LogLevel::Error;
    if (level == "off") return LogLevel::Off;
    return LogLevel::Info;
}

Status apply_request_config(TaskRequest &request, const ModelConfig &config) {
    request.sampling.seed = config.parsed_run_config.sampling.seed;
    request.sampling.repeat_last_n = config.parsed_run_config.sampling.repeat_last_n;

    const size_t max_input = config.parsed_run_config.limits.max_input_bytes;
    const size_t max_multimodal = config.parsed_run_config.limits.max_multimodal_bytes;
    size_t input_bytes = 0;
    size_t multimodal_bytes = 0;
    auto add_size = [](size_t &total, size_t amount) {
        if (amount > std::numeric_limits<size_t>::max() - total) {
            total = std::numeric_limits<size_t>::max();
        } else {
            total += amount;
        }
    };
    auto add_count = [&](size_t &total, size_t count, size_t element_size) {
        if (count > std::numeric_limits<size_t>::max() / element_size) {
            total = std::numeric_limits<size_t>::max();
        } else {
            add_size(total, count * element_size);
        }
    };

    if (const auto *input = std::get_if<PromptTextInput>(&request.input)) {
        add_size(input_bytes, input->text.size());
    } else if (const auto *input = std::get_if<PromptTokensInput>(&request.input)) {
        add_count(input_bytes, input->tokens.size(), sizeof(int));
    } else if (const auto *input = std::get_if<ChatInput>(&request.input)) {
        for (const auto &message : input->history) {
            add_size(input_bytes, message.system.size());
            add_size(input_bytes, message.query.size());
            add_size(input_bytes, message.response.size());
        }
    } else if (const auto *input = std::get_if<LegacyMultiModalInput>(&request.input)) {
        for (const auto &item : input->items) {
            add_size(input_bytes, item.prompt_text.size());
            add_size(multimodal_bytes, item.feature.size());
        }
    } else if (const auto *input = std::get_if<MultiModalChatInput>(&request.input)) {
        for (const auto &message : input->history) {
            add_size(input_bytes, message.role.size());
            for (const auto &content : message.contents) {
                add_size(input_bytes, content.type.size());
                add_size(input_bytes, content.text_data.size());
                add_size(input_bytes, content.cache_key.size());
                add_size(input_bytes, content.ext.language.size());
                add_size(multimodal_bytes, content.binary_data.size());
                add_count(multimodal_bytes, content.ext.frame_index.size(), sizeof(int));
            }
        }
    }
    for (const auto &choice : request.guided_choices) add_size(input_bytes, choice.size());
    for (const auto &tool : request.tools) add_size(input_bytes, tool.dump().size());
    add_size(input_bytes, request.visual_encoder_id.size());

    if (max_input > 0 && input_bytes > max_input) {
        return Status::invalid_argument("Task input exceeds configured max_input_bytes");
    }
    if (max_multimodal > 0 && multimodal_bytes > max_multimodal) {
        return Status::invalid_argument("Task media exceeds configured max_multimodal_bytes");
    }
    return Status::ok();
}

} // namespace

extern "C" {

void inferlet_init(void) {
    clear_last_error();
    try {
        Runtime::instance().init();
    } catch (const std::exception &e) {
        log_error(std::string("inferlet_init exception: ") + e.what());
    } catch (...) {
        log_error("inferlet_init unknown exception");
    }
}

void inferlet_deinit(void) {
    clear_last_error();
    try {
        Runtime::instance().deinit();
    } catch (const std::exception &e) {
        log_error(std::string("inferlet_deinit exception: ") + e.what());
    } catch (...) {
        log_error("inferlet_deinit unknown exception");
    }
}

void *inferlet_load_from_file(const char *model_path, void *param) {
    clear_last_error();
    try {
        if (!Runtime::instance().is_initialized()) {
            publish_error(Status::invalid_state("Inferlet is not initialized; call inferlet_init first"));
            return nullptr;
        }
        if (!model_path || std::strlen(model_path) == 0) {
            publish_error(Status::invalid_argument("model_path cannot be NULL or empty"));
            return nullptr;
        }
        ModelConfig config;
        if (param) {
            auto *p = checked_cast<ParameterInternal>(param, ObjectKind::Parameter);
            if (!p) {
                publish_error(Status::invalid_handle("Invalid Parameter handle"));
                return nullptr;
            }
            if (!p->config.valid) {
                publish_error(Status::invalid_argument("Invalid parameter configuration: " + p->config.invalid_reason));
                return nullptr;
            }
            Status merge_st = p->config.load_and_merge_run_config();
            if (!merge_st) {
                publish_error(merge_st);
                return nullptr;
            }
            config = p->config;
        }

        Status supported_st = config.validate_supported();
        if (!supported_st) {
            publish_error(supported_st);
            return nullptr;
        }
        set_log_level(configured_log_level(config.parsed_run_config.logging.level));

        std::shared_ptr<SharedModel> shared_model;
        Status load_st = ModelRegistry::instance().get_or_load(model_path, config, shared_model);
        if (!load_st) {
            publish_error(load_st);
            return nullptr;
        }

        auto reg_st = Runtime::instance().register_handle();
        if (!reg_st) {
            ModelRegistry::instance().release(shared_model);
            publish_error(reg_st);
            return nullptr;
        }

        auto *handle = new (std::nothrow) ModelHandleInternal(shared_model);
        if (!handle) {
            Runtime::instance().unregister_handle();
            ModelRegistry::instance().release(shared_model);
            publish_error(Status::out_of_memory("Failed to allocate ModelHandle"));
            return nullptr;
        }
        LiveObjectRegistry::instance().register_object(handle, ObjectKind::ModelHandle);
        return handle;
    } catch (const std::bad_alloc &) {
        publish_error(Status::out_of_memory());
        return nullptr;
    } catch (const std::exception &e) {
        publish_error(Status::internal(e.what()));
        return nullptr;
    } catch (...) {
        publish_error(Status::internal());
        return nullptr;
    }
}

void inferlet_unload(void *handle) {
    clear_last_error();
    if (!handle) return;
    try {
        auto *h = checked_cast<ModelHandleInternal>(handle, ObjectKind::ModelHandle);
        if (!h) {
            log_warn("inferlet_unload called with invalid handle");
            return;
        }

        std::vector<std::shared_ptr<TaskCore>> tasks_to_cancel;
        std::vector<TaskTicketInternal *> tickets_to_delete;
        std::vector<StreamTicketInternal *> stream_tickets_to_delete;
        {
            std::lock_guard<std::mutex> lock(h->task_mutex);
            h->closing = true;
            tasks_to_cancel.assign(h->tasks.begin(), h->tasks.end());
            tickets_to_delete.assign(h->task_tickets.begin(), h->task_tickets.end());
            stream_tickets_to_delete.assign(h->stream_tickets.begin(), h->stream_tickets.end());
        }

        for (auto &t : tasks_to_cancel) {
            t->cancel_requested.store(true);
            t->cv.notify_all();
            t->queue_not_full.notify_all();
        }

        for (auto &t : tasks_to_cancel) {
            std::unique_lock<std::mutex> tlock(t->mutex);
            t->cv.wait(tlock, [&]() { return t->is_terminal(); });
        }

        for (auto *tick : tickets_to_delete) {
            LiveObjectRegistry::instance().unregister_object(tick);
            delete tick;
        }
        for (auto *stick : stream_tickets_to_delete) {
            LiveObjectRegistry::instance().unregister_object(stick);
            delete stick;
        }

        {
            std::lock_guard<std::mutex> lock(h->task_mutex);
            h->tasks.clear();
            h->task_tickets.clear();
            h->stream_tickets.clear();
        }

        LiveObjectRegistry::instance().unregister_object(h);
        auto shared = h->shared_model;
        delete h;

        if (shared) {
            ModelRegistry::instance().release(shared);
        }
        Runtime::instance().unregister_handle();
    } catch (const std::exception &e) {
        log_error(std::string("inferlet_unload exception: ") + e.what());
    } catch (...) {
        log_error("inferlet_unload unknown exception");
    }
}

int inferlet_run(void *handle, void *task_in, void **task_out) {
    clear_last_error();
    if (task_out) *task_out = nullptr;
    try {
        if (!handle) return publish_error(Status::invalid_handle("handle cannot be NULL"));
        if (!task_in) return publish_error(Status::invalid_argument("task_in cannot be NULL"));
        if (!task_out) return publish_error(Status::invalid_argument("task_out cannot be NULL"));

        auto *h = checked_cast<ModelHandleInternal>(handle, ObjectKind::ModelHandle);
        if (!h) return publish_error(Status::invalid_handle("Invalid ModelHandle"));
        if (h->is_tokenizer_only()) return publish_error(Status::unsupported("Model was loaded in tokenizer-only mode"));

        Status ready_st = h->wait_for_ready();
        if (!ready_st) return publish_error(ready_st);

        auto *in = checked_cast<TaskInputInternal>(task_in, ObjectKind::TaskInput);
        if (!in) return publish_error(Status::invalid_handle("Invalid TaskInput handle"));

        TaskRequest req;
        Status st = in->create_request(req);
        if (!st) return publish_error(st);
        st = apply_request_config(req, h->shared_model->config);
        if (!st) return publish_error(st);

        SyncExecutionGuard execution(h->shared_model);
        auto out = std::make_unique<TaskOutputInternal>();

        CollectingSink sink(*out, h->shared_model->config.parsed_run_config.limits.max_verbose_bytes);
        st = Generator::run(req, h->shared_model, sink);
        if (st.is_ok() && !sink.finish_status().is_ok()) {
            st = sink.finish_status();
        }
        if (!st) {
            return publish_error(st);
        }

        LiveObjectRegistry::instance().register_object(out.get(), ObjectKind::TaskOutput);
        *task_out = out.release();
        return INFERLET_OK;
    } catch (const std::bad_alloc &) {
        return publish_error(Status::out_of_memory());
    } catch (const std::exception &e) {
        return publish_error(Status::internal(e.what()));
    } catch (...) {
        return publish_error(Status::internal());
    }
}

int inferlet_run_async(void *handle, void *task_in, void **task) {
    clear_last_error();
    if (task) *task = nullptr;
    try {
        if (!handle) return publish_error(Status::invalid_handle("handle cannot be NULL"));
        if (!task_in) return publish_error(Status::invalid_argument("task_in cannot be NULL"));
        if (!task) return publish_error(Status::invalid_argument("task cannot be NULL"));

        auto *h = checked_cast<ModelHandleInternal>(handle, ObjectKind::ModelHandle);
        if (!h) return publish_error(Status::invalid_handle("Invalid ModelHandle"));
        if (h->is_tokenizer_only()) return publish_error(Status::unsupported("Model was loaded in tokenizer-only mode"));

        auto *in = checked_cast<TaskInputInternal>(task_in, ObjectKind::TaskInput);
        if (!in) return publish_error(Status::invalid_handle("Invalid TaskInput handle"));

        TaskRequest req;
        Status st = in->create_request(req);
        if (!st) return publish_error(st);
        st = apply_request_config(req, h->shared_model->config);
        if (!st) return publish_error(st);

        auto core = std::make_shared<TaskCore>();
        core->request = std::move(req);
        core->model = h->shared_model;
        core->streaming = false;

        auto *ticket = new (std::nothrow) TaskTicketInternal();
        if (!ticket) return publish_error(Status::out_of_memory());
        ticket->core = core;
        ticket->owner = h;

        LiveObjectRegistry::instance().register_object(ticket, ObjectKind::TaskTicket);

        {
            std::lock_guard<std::mutex> lock(h->task_mutex);
            if (h->closing) {
                LiveObjectRegistry::instance().unregister_object(ticket);
                delete ticket;
                return publish_error(Status::invalid_state("ModelHandle is closing"));
            }
            h->tasks.insert(core);
            h->task_tickets.insert(ticket);
        }

        st = Runtime::instance().scheduler().submit(core);
        if (!st) {
            {
                std::lock_guard<std::mutex> lock(h->task_mutex);
                h->tasks.erase(core);
                h->task_tickets.erase(ticket);
            }
            LiveObjectRegistry::instance().unregister_object(ticket);
            delete ticket;
            return publish_error(st);
        }

        *task = ticket;
        return INFERLET_OK;
    } catch (const std::bad_alloc &) {
        return publish_error(Status::out_of_memory());
    } catch (const std::exception &e) {
        return publish_error(Status::internal(e.what()));
    } catch (...) {
        return publish_error(Status::internal());
    }
}

int inferlet_run_stream(void *handle, void *task_in, void **task_stream) {
    clear_last_error();
    if (task_stream) *task_stream = nullptr;
    try {
        if (!handle) return publish_error(Status::invalid_handle("handle cannot be NULL"));
        if (!task_in) return publish_error(Status::invalid_argument("task_in cannot be NULL"));
        if (!task_stream) return publish_error(Status::invalid_argument("task_stream cannot be NULL"));

        auto *h = checked_cast<ModelHandleInternal>(handle, ObjectKind::ModelHandle);
        if (!h) return publish_error(Status::invalid_handle("Invalid ModelHandle"));
        if (h->is_tokenizer_only()) return publish_error(Status::unsupported("Model was loaded in tokenizer-only mode"));

        auto *in = checked_cast<TaskInputInternal>(task_in, ObjectKind::TaskInput);
        if (!in) return publish_error(Status::invalid_handle("Invalid TaskInput handle"));

        TaskRequest req;
        Status st = in->create_request(req);
        if (!st) return publish_error(st);
        st = apply_request_config(req, h->shared_model->config);
        if (!st) return publish_error(st);

        auto core = std::make_shared<TaskCore>();
        core->request = std::move(req);
        core->model = h->shared_model;
        core->streaming = true;
        core->max_queued_chunks = 16;
        if (h->shared_model && h->shared_model->config.parsed_run_config.scheduler.stream_queue_chunks > 0) {
            core->max_queued_chunks = h->shared_model->config.parsed_run_config.scheduler.stream_queue_chunks;
        }

        auto *ticket = new (std::nothrow) StreamTicketInternal();
        if (!ticket) return publish_error(Status::out_of_memory());
        ticket->core = core;
        ticket->owner = h;

        LiveObjectRegistry::instance().register_object(ticket, ObjectKind::StreamTicket);

        {
            std::lock_guard<std::mutex> lock(h->task_mutex);
            if (h->closing) {
                LiveObjectRegistry::instance().unregister_object(ticket);
                delete ticket;
                return publish_error(Status::invalid_state("ModelHandle is closing"));
            }
            h->tasks.insert(core);
            h->stream_tickets.insert(ticket);
        }

        st = Runtime::instance().scheduler().submit(core);
        if (!st) {
            {
                std::lock_guard<std::mutex> lock(h->task_mutex);
                h->tasks.erase(core);
                h->stream_tickets.erase(ticket);
            }
            LiveObjectRegistry::instance().unregister_object(ticket);
            delete ticket;
            return publish_error(st);
        }

        *task_stream = ticket;
        return INFERLET_OK;
    } catch (const std::bad_alloc &) {
        return publish_error(Status::out_of_memory());
    } catch (const std::exception &e) {
        return publish_error(Status::internal(e.what()));
    } catch (...) {
        return publish_error(Status::internal());
    }
}

int inferlet_wait_task(void *handle, void *task, void **task_out) {
    clear_last_error();
    if (task_out) *task_out = nullptr;
    try {
        if (!handle) return publish_error(Status::invalid_handle("handle cannot be NULL"));
        if (!task) return publish_error(Status::invalid_handle("task cannot be NULL"));
        if (!task_out) return publish_error(Status::invalid_argument("task_out cannot be NULL"));

        auto *h = checked_cast<ModelHandleInternal>(handle, ObjectKind::ModelHandle);
        if (!h) return publish_error(Status::invalid_handle("Invalid ModelHandle"));

        auto *t = static_cast<TaskTicketInternal *>(task);
        std::shared_ptr<TaskCore> core;
        {
            std::lock_guard<std::mutex> hlock(h->task_mutex);
            if (h->task_tickets.find(t) == h->task_tickets.end()) {
                return publish_error(Status::invalid_handle("Invalid TaskTicket"));
            }
            if (t->owner != h) {
                return publish_error(Status::invalid_handle("TaskTicket does not belong to this handle"));
            }
            if (t->consuming) {
                return publish_error(Status::invalid_state("TaskTicket is already being consumed"));
            }
            t->consuming = true;
            core = t->core;
        }
        if (!core) {
            {
                std::lock_guard<std::mutex> hlock(h->task_mutex);
                h->task_tickets.erase(t);
            }
            LiveObjectRegistry::instance().unregister_object(t);
            delete t;
            return publish_error(Status::invalid_handle("TaskTicket has null core"));
        }

        std::unique_lock<std::mutex> lock(core->mutex);
        core->cv.wait(lock, [&]() {
            return core->is_terminal();
        });

        TaskState final_state = core->state;
        Status final_status = core->terminal_status;
        auto output = std::move(core->output);
        core->state = TaskState::Consumed;
        lock.unlock();

        {
            std::lock_guard<std::mutex> hlock(h->task_mutex);
            h->tasks.erase(core);
            h->task_tickets.erase(t);
        }

        LiveObjectRegistry::instance().unregister_object(t);
        delete t;

        if (final_state == TaskState::Completed && output) {
            TaskOutputInternal *raw_out = output.release();
            LiveObjectRegistry::instance().register_object(raw_out, ObjectKind::TaskOutput);
            *task_out = raw_out;
            return INFERLET_OK;
        } else if (final_state == TaskState::Cancelled) {
            return publish_error(Status::cancelled());
        } else {
            return publish_error(final_status.is_ok() ? Status::inference_error() : final_status);
        }
    } catch (const std::bad_alloc &) {
        return publish_error(Status::out_of_memory());
    } catch (const std::exception &e) {
        return publish_error(Status::internal(e.what()));
    } catch (...) {
        return publish_error(Status::internal());
    }
}

int inferlet_query_task(void *handle, void *task, void **task_out) {
    clear_last_error();
    if (task_out) *task_out = nullptr;
    try {
        if (!handle) return publish_error(Status::invalid_handle("handle cannot be NULL"));
        if (!task) return publish_error(Status::invalid_handle("task cannot be NULL"));
        if (!task_out) return publish_error(Status::invalid_argument("task_out cannot be NULL"));

        auto *h = checked_cast<ModelHandleInternal>(handle, ObjectKind::ModelHandle);
        if (!h) return publish_error(Status::invalid_handle("Invalid ModelHandle"));

        auto *t = static_cast<TaskTicketInternal *>(task);
        std::shared_ptr<TaskCore> core;
        {
            std::lock_guard<std::mutex> hlock(h->task_mutex);
            if (h->task_tickets.find(t) == h->task_tickets.end()) {
                return publish_error(Status::invalid_handle("Invalid TaskTicket"));
            }
            if (t->owner != h) {
                return publish_error(Status::invalid_handle("TaskTicket does not belong to this handle"));
            }
            if (t->consuming) {
                return publish_error(Status::invalid_state("TaskTicket is already being consumed"));
            }
            t->consuming = true;
            core = t->core;
        }
        if (!core) {
            {
                std::lock_guard<std::mutex> hlock(h->task_mutex);
                h->task_tickets.erase(t);
            }
            LiveObjectRegistry::instance().unregister_object(t);
            delete t;
            return publish_error(Status::invalid_handle("TaskTicket has null core"));
        }

        std::unique_lock<std::mutex> lock(core->mutex);
        if (!core->is_terminal()) {
            lock.unlock();
            std::lock_guard<std::mutex> hlock(h->task_mutex);
            if (h->task_tickets.find(t) != h->task_tickets.end()) {
                t->consuming = false;
            }
            *task_out = nullptr;
            return INFERLET_OK;
        }

        TaskState final_state = core->state;
        Status final_status = core->terminal_status;
        auto output = std::move(core->output);
        core->state = TaskState::Consumed;
        lock.unlock();

        {
            std::lock_guard<std::mutex> hlock(h->task_mutex);
            h->tasks.erase(core);
            h->task_tickets.erase(t);
        }

        LiveObjectRegistry::instance().unregister_object(t);
        delete t;

        if (final_state == TaskState::Completed && output) {
            TaskOutputInternal *raw_out = output.release();
            LiveObjectRegistry::instance().register_object(raw_out, ObjectKind::TaskOutput);
            *task_out = raw_out;
            return INFERLET_OK;
        } else if (final_state == TaskState::Cancelled) {
            return publish_error(Status::cancelled());
        } else {
            return publish_error(final_status.is_ok() ? Status::inference_error() : final_status);
        }
    } catch (const std::bad_alloc &) {
        return publish_error(Status::out_of_memory());
    } catch (const std::exception &e) {
        return publish_error(Status::internal(e.what()));
    } catch (...) {
        return publish_error(Status::internal());
    }
}

int inferlet_cancel_task(void *handle, void *task) {
    clear_last_error();
    try {
        if (!handle) return publish_error(Status::invalid_handle("handle cannot be NULL"));
        if (!task) return publish_error(Status::invalid_handle("task cannot be NULL"));

        auto *h = checked_cast<ModelHandleInternal>(handle, ObjectKind::ModelHandle);
        if (!h) return publish_error(Status::invalid_handle("Invalid ModelHandle"));

        std::shared_ptr<TaskCore> core;
        {
            std::lock_guard<std::mutex> hlock(h->task_mutex);
            auto *t = static_cast<TaskTicketInternal *>(task);
            auto ticket_it = h->task_tickets.find(t);
            if (ticket_it != h->task_tickets.end()) {
                core = t->core;
            }
        }
        if (core) {
            core->cancel_requested.store(true);
            {
                std::lock_guard<std::mutex> lock(core->mutex);
                if (core->state == TaskState::Running) {
                    core->state = TaskState::Cancelling;
                }
            }
            core->cv.notify_all();
            core->queue_not_full.notify_all();
            return INFERLET_OK;
        }

        auto *st = checked_cast<StreamTicketInternal>(task, ObjectKind::StreamTicket);
        if (st) {
            if (st->owner != h) {
                return publish_error(Status::invalid_handle("StreamTicket does not belong to this handle"));
            }
            auto core = st->core;
            if (core) {
                core->cancel_requested.store(true);
                core->cv.notify_all();
                core->queue_not_full.notify_all();
                std::unique_lock<std::mutex> lock(core->mutex);
                core->cv.wait(lock, [&]() { return core->is_terminal(); });
                core->chunks.clear();
                core->current_chunk.reset();
                core->state = TaskState::Consumed;
            }
            {
                std::lock_guard<std::mutex> hlock(h->task_mutex);
                h->tasks.erase(core);
                h->stream_tickets.erase(st);
            }
            LiveObjectRegistry::instance().unregister_object(st);
            delete st;
            return INFERLET_OK;
        }

        return publish_error(Status::invalid_handle("Invalid task handle"));
    } catch (const std::bad_alloc &) {
        return publish_error(Status::out_of_memory());
    } catch (const std::exception &e) {
        return publish_error(Status::internal(e.what()));
    } catch (...) {
        return publish_error(Status::internal());
    }
}

int inferlet_check_backend_active(void *handle, int *is_active) {
    clear_last_error();
    if (is_active) *is_active = 0;
    try {
        if (!handle) return publish_error(Status::invalid_handle("handle cannot be NULL"));
        if (!is_active) return publish_error(Status::invalid_argument("is_active cannot be NULL"));

        auto *h = checked_cast<ModelHandleInternal>(handle, ObjectKind::ModelHandle);
        if (!h) return publish_error(Status::invalid_handle("Invalid ModelHandle"));

        if (h->shared_model) {
            std::lock_guard<std::mutex> sm_lock(h->shared_model->mutex);
            if (h->shared_model->load_state == LoadState::Failed) {
                Status err = h->shared_model->load_status;
                if (err.is_ok() || err.code != INFERLET_E_MODEL_LOAD) {
                    err = Status::model_load(err.message.empty() ? "Model load failed" : err.message);
                }
                return publish_error(err);
            }
            if (h->shared_model->load_state == LoadState::Loading) {
                *is_active = 1;
                return INFERLET_OK;
            }
            if (h->shared_model->queued.load() > 0 || h->shared_model->running.load() > 0) {
                *is_active = 1;
                return INFERLET_OK;
            }
        }
        // Unconsumed terminal tickets retain results, not backend work.
        *is_active = 0;
        return INFERLET_OK;
    } catch (...) {
        return publish_error(Status::internal());
    }
}

// =================== TaskInput APIs ===================

void *inferlet_task_input_allocate(void) {
    clear_last_error();
    try {
        auto *input = new (std::nothrow) TaskInputInternal();
        if (!input) {
            publish_error(Status::out_of_memory("Failed to allocate TaskInput"));
            return nullptr;
        }
        LiveObjectRegistry::instance().register_object(input, ObjectKind::TaskInput);
        return input;
    } catch (const std::bad_alloc &) {
        publish_error(Status::out_of_memory());
        return nullptr;
    } catch (const std::exception &e) {
        publish_error(Status::internal(e.what()));
        return nullptr;
    } catch (...) {
        publish_error(Status::internal());
        return nullptr;
    }
}

void inferlet_task_input_deallocate(void *task_in) {
    clear_last_error();
    if (!task_in) return;
    try {
        auto *in = checked_cast<TaskInputInternal>(task_in, ObjectKind::TaskInput);
        if (!in) {
            log_warn("inferlet_task_input_deallocate called with invalid handle");
            return;
        }
        LiveObjectRegistry::instance().unregister_object(in);
        delete in;
    } catch (const std::exception &e) {
        log_error(std::string("inferlet_task_input_deallocate exception: ") + e.what());
    } catch (...) {
        log_error("inferlet_task_input_deallocate unknown exception");
    }
}

int inferlet_task_input_set_prompt_text(void *task_in, const char *prompt_text) {
    clear_last_error();
    try {
        auto *in = checked_cast<TaskInputInternal>(task_in, ObjectKind::TaskInput);
        if (!in) return publish_error(Status::invalid_handle("Invalid TaskInput handle"));
        return publish_status(in->set_prompt_text(prompt_text));
    } catch (const std::bad_alloc &) {
        return publish_error(Status::out_of_memory());
    } catch (const std::exception &e) {
        return publish_error(Status::internal(e.what()));
    } catch (...) {
        return publish_error(Status::internal());
    }
}

int inferlet_task_input_set_prompt_tokens(void *task_in, int *prompt_tokens, int prompt_token_length) {
    clear_last_error();
    try {
        auto *in = checked_cast<TaskInputInternal>(task_in, ObjectKind::TaskInput);
        if (!in) return publish_error(Status::invalid_handle("Invalid TaskInput handle"));
        return publish_status(in->set_prompt_tokens(prompt_tokens, prompt_token_length));
    } catch (const std::bad_alloc &) {
        return publish_error(Status::out_of_memory());
    } catch (const std::exception &e) {
        return publish_error(Status::internal(e.what()));
    } catch (...) {
        return publish_error(Status::internal());
    }
}

int inferlet_task_input_set_chat_history(void *task_in, inferlet_chat_history chat_history[], int history_length) {
    clear_last_error();
    try {
        auto *in = checked_cast<TaskInputInternal>(task_in, ObjectKind::TaskInput);
        if (!in) return publish_error(Status::invalid_handle("Invalid TaskInput handle"));
        return publish_status(in->set_chat_history(chat_history, history_length));
    } catch (const std::bad_alloc &) {
        return publish_error(Status::out_of_memory());
    } catch (const std::exception &e) {
        return publish_error(Status::internal(e.what()));
    } catch (...) {
        return publish_error(Status::internal());
    }
}

int inferlet_task_input_set_multimodal(void *task_in, inferlet_multimodal mm[], int mm_length) {
    clear_last_error();
    try {
        auto *in = checked_cast<TaskInputInternal>(task_in, ObjectKind::TaskInput);
        if (!in) return publish_error(Status::invalid_handle("Invalid TaskInput handle"));
        return publish_status(in->set_multimodal(mm, mm_length));
    } catch (const std::bad_alloc &) {
        return publish_error(Status::out_of_memory());
    } catch (const std::exception &e) {
        return publish_error(Status::internal(e.what()));
    } catch (...) {
        return publish_error(Status::internal());
    }
}

int inferlet_task_input_set_multimodal_chat_history(void *task_in, inferlet_multimodal_chat_history chat_history[], int history_length) {
    clear_last_error();
    try {
        auto *in = checked_cast<TaskInputInternal>(task_in, ObjectKind::TaskInput);
        if (!in) return publish_error(Status::invalid_handle("Invalid TaskInput handle"));
        return publish_status(in->set_multimodal_chat(chat_history, history_length));
    } catch (const std::bad_alloc &) {
        return publish_error(Status::out_of_memory());
    } catch (const std::exception &e) {
        return publish_error(Status::internal(e.what()));
    } catch (...) {
        return publish_error(Status::internal());
    }
}

int inferlet_task_input_set_low_rank_adapter_id(void *task_in, int lora_adapter_id) {
    clear_last_error();
    try {
        auto *in = checked_cast<TaskInputInternal>(task_in, ObjectKind::TaskInput);
        if (!in) return publish_error(Status::invalid_handle("Invalid TaskInput handle"));
        if (lora_adapter_id < 0) return publish_error(Status::invalid_argument("lora_adapter_id must be >= 0"));
        in->lora_id = lora_adapter_id;
        return INFERLET_OK;
    } catch (const std::bad_alloc &) {
        return publish_error(Status::out_of_memory());
    } catch (const std::exception &e) {
        return publish_error(Status::internal(e.what()));
    } catch (...) {
        return publish_error(Status::internal());
    }
}

int inferlet_task_input_set_beam_size(void *task_in, int beam_size) {
    clear_last_error();
    try {
        auto *in = checked_cast<TaskInputInternal>(task_in, ObjectKind::TaskInput);
        if (!in) return publish_error(Status::invalid_handle("Invalid TaskInput handle"));
        return publish_status(in->set_beam_size(beam_size));
    } catch (const std::bad_alloc &) {
        return publish_error(Status::out_of_memory());
    } catch (const std::exception &e) {
        return publish_error(Status::internal(e.what()));
    } catch (...) {
        return publish_error(Status::internal());
    }
}

int inferlet_task_input_set_top_k(void *task_in, int top_k) {
    clear_last_error();
    try {
        auto *in = checked_cast<TaskInputInternal>(task_in, ObjectKind::TaskInput);
        if (!in) return publish_error(Status::invalid_handle("Invalid TaskInput handle"));
        return publish_status(in->set_top_k(top_k));
    } catch (const std::bad_alloc &) {
        return publish_error(Status::out_of_memory());
    } catch (const std::exception &e) {
        return publish_error(Status::internal(e.what()));
    } catch (...) {
        return publish_error(Status::internal());
    }
}

int inferlet_task_input_set_top_p(void *task_in, float top_p) {
    clear_last_error();
    try {
        auto *in = checked_cast<TaskInputInternal>(task_in, ObjectKind::TaskInput);
        if (!in) return publish_error(Status::invalid_handle("Invalid TaskInput handle"));
        return publish_status(in->set_top_p(top_p));
    } catch (const std::bad_alloc &) {
        return publish_error(Status::out_of_memory());
    } catch (const std::exception &e) {
        return publish_error(Status::internal(e.what()));
    } catch (...) {
        return publish_error(Status::internal());
    }
}

int inferlet_task_input_set_temperature(void *task_in, float temperature) {
    clear_last_error();
    try {
        auto *in = checked_cast<TaskInputInternal>(task_in, ObjectKind::TaskInput);
        if (!in) return publish_error(Status::invalid_handle("Invalid TaskInput handle"));
        return publish_status(in->set_temperature(temperature));
    } catch (const std::bad_alloc &) {
        return publish_error(Status::out_of_memory());
    } catch (const std::exception &e) {
        return publish_error(Status::internal(e.what()));
    } catch (...) {
        return publish_error(Status::internal());
    }
}

int inferlet_task_input_set_repetition_penalty(void *task_in, float repetition_penalty) {
    clear_last_error();
    try {
        auto *in = checked_cast<TaskInputInternal>(task_in, ObjectKind::TaskInput);
        if (!in) return publish_error(Status::invalid_handle("Invalid TaskInput handle"));
        return publish_status(in->set_repetition_penalty(repetition_penalty));
    } catch (const std::bad_alloc &) {
        return publish_error(Status::out_of_memory());
    } catch (const std::exception &e) {
        return publish_error(Status::internal(e.what()));
    } catch (...) {
        return publish_error(Status::internal());
    }
}

int inferlet_task_input_set_max_output_tokens(void *task_in, int max_output_tokens) {
    clear_last_error();
    try {
        auto *in = checked_cast<TaskInputInternal>(task_in, ObjectKind::TaskInput);
        if (!in) return publish_error(Status::invalid_handle("Invalid TaskInput handle"));
        return publish_status(in->set_max_output_tokens(max_output_tokens));
    } catch (const std::bad_alloc &) {
        return publish_error(Status::out_of_memory());
    } catch (const std::exception &e) {
        return publish_error(Status::internal(e.what()));
    } catch (...) {
        return publish_error(Status::internal());
    }
}

int inferlet_task_input_set_probability_type(void *task_in, int probability_type) {
    clear_last_error();
    try {
        auto *in = checked_cast<TaskInputInternal>(task_in, ObjectKind::TaskInput);
        if (!in) return publish_error(Status::invalid_handle("Invalid TaskInput handle"));
        return publish_status(in->set_probability_type(probability_type));
    } catch (const std::bad_alloc &) {
        return publish_error(Status::out_of_memory());
    } catch (const std::exception &e) {
        return publish_error(Status::internal(e.what()));
    } catch (...) {
        return publish_error(Status::internal());
    }
}

int inferlet_task_input_set_verbose_data_flag(void *task_in, int verbose_flag) {
    clear_last_error();
    try {
        auto *in = checked_cast<TaskInputInternal>(task_in, ObjectKind::TaskInput);
        if (!in) return publish_error(Status::invalid_handle("Invalid TaskInput handle"));
        return publish_status(in->set_verbose_flag(verbose_flag));
    } catch (const std::bad_alloc &) {
        return publish_error(Status::out_of_memory());
    } catch (const std::exception &e) {
        return publish_error(Status::internal(e.what()));
    } catch (...) {
        return publish_error(Status::internal());
    }
}

int inferlet_task_input_set_stream_block_mode(void *task_in, int stream_block) {
    clear_last_error();
    try {
        auto *in = checked_cast<TaskInputInternal>(task_in, ObjectKind::TaskInput);
        if (!in) return publish_error(Status::invalid_handle("Invalid TaskInput handle"));
        return publish_status(in->set_stream_block(stream_block));
    } catch (const std::bad_alloc &) {
        return publish_error(Status::out_of_memory());
    } catch (const std::exception &e) {
        return publish_error(Status::internal(e.what()));
    } catch (...) {
        return publish_error(Status::internal());
    }
}

int inferlet_task_input_set_thinking_mode(void *task_in, int thinking) {
    clear_last_error();
    try {
        auto *in = checked_cast<TaskInputInternal>(task_in, ObjectKind::TaskInput);
        if (!in) return publish_error(Status::invalid_handle("Invalid TaskInput handle"));
        return publish_status(in->set_thinking(thinking));
    } catch (const std::bad_alloc &) {
        return publish_error(Status::out_of_memory());
    } catch (const std::exception &e) {
        return publish_error(Status::internal(e.what()));
    } catch (...) {
        return publish_error(Status::internal());
    }
}

int inferlet_task_input_set_special_token_mode(void *task_in, int keep_special_token) {
    clear_last_error();
    try {
        auto *in = checked_cast<TaskInputInternal>(task_in, ObjectKind::TaskInput);
        if (!in) return publish_error(Status::invalid_handle("Invalid TaskInput handle"));
        return publish_status(in->set_special_token(keep_special_token));
    } catch (const std::bad_alloc &) {
        return publish_error(Status::out_of_memory());
    } catch (const std::exception &e) {
        return publish_error(Status::internal(e.what()));
    } catch (...) {
        return publish_error(Status::internal());
    }
}

int inferlet_task_input_set_guided_choices(void *task_in, const char *guided_choices[], int guided_choices_num) {
    clear_last_error();
    try {
        auto *in = checked_cast<TaskInputInternal>(task_in, ObjectKind::TaskInput);
        if (!in) return publish_error(Status::invalid_handle("Invalid TaskInput handle"));
        return publish_status(in->set_guided_choices(guided_choices, guided_choices_num));
    } catch (const std::bad_alloc &) {
        return publish_error(Status::out_of_memory());
    } catch (const std::exception &e) {
        return publish_error(Status::internal(e.what()));
    } catch (...) {
        return publish_error(Status::internal());
    }
}

int inferlet_task_input_set_tools(void *task_in, const char *tools[], int tools_num) {
    clear_last_error();
    try {
        auto *in = checked_cast<TaskInputInternal>(task_in, ObjectKind::TaskInput);
        if (!in) return publish_error(Status::invalid_handle("Invalid TaskInput handle"));
        return publish_status(in->set_tools(tools, tools_num));
    } catch (const std::bad_alloc &) {
        return publish_error(Status::out_of_memory());
    } catch (const std::exception &e) {
        return publish_error(Status::internal(e.what()));
    } catch (...) {
        return publish_error(Status::internal());
    }
}

int inferlet_task_input_set_visual_encoder(void *task_in, const char *visual_encoder_id) {
    clear_last_error();
    try {
        auto *in = checked_cast<TaskInputInternal>(task_in, ObjectKind::TaskInput);
        if (!in) return publish_error(Status::invalid_handle("Invalid TaskInput handle"));
        return publish_status(in->set_visual_encoder(visual_encoder_id));
    } catch (const std::bad_alloc &) {
        return publish_error(Status::out_of_memory());
    } catch (const std::exception &e) {
        return publish_error(Status::internal(e.what()));
    } catch (...) {
        return publish_error(Status::internal());
    }
}

// =================== TaskOutput APIs ===================

void inferlet_task_output_deallocate(void *task_out) {
    clear_last_error();
    if (!task_out) return;
    try {
        auto *out = checked_cast<TaskOutputInternal>(task_out, ObjectKind::TaskOutput);
        if (!out) {
            log_warn("inferlet_task_output_deallocate called with invalid handle");
            return;
        }
        LiveObjectRegistry::instance().unregister_object(out);
        delete out;
    } catch (const std::exception &e) {
        log_error(std::string("inferlet_task_output_deallocate exception: ") + e.what());
    } catch (...) {
        log_error("inferlet_task_output_deallocate unknown exception");
    }
}

const char *inferlet_task_output_get_result_text(void *task_out) {
    clear_last_error();
    try {
        auto *out = checked_cast<TaskOutputInternal>(task_out, ObjectKind::TaskOutput);
        if (!out) {
            publish_error(Status::invalid_handle("Invalid TaskOutput handle"));
            return nullptr;
        }
        return out->result.text.c_str();
    } catch (...) {
        publish_error(Status::internal());
        return nullptr;
    }
}

const int *inferlet_task_output_get_result_tokens(void *task_out, int *real_token_length) {
    clear_last_error();
    if (real_token_length) *real_token_length = 0;
    try {
        auto *out = checked_cast<TaskOutputInternal>(task_out, ObjectKind::TaskOutput);
        if (!out) {
            publish_error(Status::invalid_handle("Invalid TaskOutput handle"));
            return nullptr;
        }
        if (out->result.tokens.empty()) {
            return nullptr;
        }
        if (real_token_length) {
            *real_token_length = static_cast<int>(out->result.tokens.size());
        }
        return out->result.tokens.data();
    } catch (...) {
        publish_error(Status::internal());
        return nullptr;
    }
}

const int *inferlet_task_output_get_token_offsets(void *task_out) {
    clear_last_error();
    try {
        auto *out = checked_cast<TaskOutputInternal>(task_out, ObjectKind::TaskOutput);
        if (!out) {
            publish_error(Status::invalid_handle("Invalid TaskOutput handle"));
            return nullptr;
        }
        if (out->result.byte_offsets.empty()) {
            return nullptr;
        }
        return out->result.byte_offsets.data();
    } catch (...) {
        publish_error(Status::internal());
        return nullptr;
    }
}

const float *inferlet_task_output_get_token_probability(void *task_out) {
    clear_last_error();
    try {
        auto *out = checked_cast<TaskOutputInternal>(task_out, ObjectKind::TaskOutput);
        if (!out) {
            publish_error(Status::invalid_handle("Invalid TaskOutput handle"));
            return nullptr;
        }
        if (out->result.probabilities.empty()) {
            return nullptr;
        }
        return out->result.probabilities.data();
    } catch (...) {
        publish_error(Status::internal());
        return nullptr;
    }
}

const int *inferlet_task_output_get_input_tokens(void *task_out, int *input_tokens_length, const char ***input_tokens_text) {
    clear_last_error();
    if (input_tokens_length) *input_tokens_length = 0;
    if (input_tokens_text) *input_tokens_text = nullptr;
    try {
        auto *out = checked_cast<TaskOutputInternal>(task_out, ObjectKind::TaskOutput);
        if (!out) {
            publish_error(Status::invalid_handle("Invalid TaskOutput handle"));
            return nullptr;
        }
        if (out->input.tokens.empty()) {
            return nullptr;
        }
        if (input_tokens_length) {
            *input_tokens_length = static_cast<int>(out->input.tokens.size());
        }
        if (input_tokens_text) {
            *input_tokens_text = out->input.text_ptrs.data();
        }
        return out->input.tokens.data();
    } catch (...) {
        publish_error(Status::internal());
        return nullptr;
    }
}

const float **inferlet_task_output_get_output_logits(void *task_out, int *output_tokens_length, int *vocab_size) {
    clear_last_error();
    if (output_tokens_length) *output_tokens_length = 0;
    if (vocab_size) *vocab_size = 0;
    try {
        auto *out = checked_cast<TaskOutputInternal>(task_out, ObjectKind::TaskOutput);
        if (!out) {
            publish_error(Status::invalid_handle("Invalid TaskOutput handle"));
            return nullptr;
        }
        if (out->result.logits.rows == 0 || out->result.logits.row_ptrs.empty()) {
            return nullptr;
        }
        if (output_tokens_length) {
            *output_tokens_length = out->result.logits.rows;
        }
        if (vocab_size) {
            *vocab_size = out->result.logits.cols;
        }
        return out->result.logits.row_ptrs.data();
    } catch (...) {
        publish_error(Status::internal());
        return nullptr;
    }
}

const float **inferlet_task_output_get_output_embedding(void *task_out, int *output_tokens_length, int *embedding_dim) {
    clear_last_error();
    if (output_tokens_length) *output_tokens_length = 0;
    if (embedding_dim) *embedding_dim = 0;
    try {
        auto *out = checked_cast<TaskOutputInternal>(task_out, ObjectKind::TaskOutput);
        if (!out) {
            publish_error(Status::invalid_handle("Invalid TaskOutput handle"));
            return nullptr;
        }
        if (out->result.embeddings.rows == 0 || out->result.embeddings.row_ptrs.empty()) {
            return nullptr;
        }
        if (output_tokens_length) {
            *output_tokens_length = out->result.embeddings.rows;
        }
        if (embedding_dim) {
            *embedding_dim = out->result.embeddings.cols;
        }
        return out->result.embeddings.row_ptrs.data();
    } catch (...) {
        publish_error(Status::internal());
        return nullptr;
    }
}

// =================== Stream APIs ===================

const char *inferlet_get_stream_text(void *handle, void *task_stream) {
    clear_last_error();
    try {
        if (!handle) {
            publish_error(Status::invalid_handle("handle cannot be NULL"));
            return nullptr;
        }
        if (!task_stream) {
            publish_error(Status::invalid_handle("task_stream cannot be NULL"));
            return nullptr;
        }
        auto *h = checked_cast<ModelHandleInternal>(handle, ObjectKind::ModelHandle);
        if (!h) {
            publish_error(Status::invalid_handle("Invalid ModelHandle"));
            return nullptr;
        }
        auto *stream = checked_cast<StreamTicketInternal>(task_stream, ObjectKind::StreamTicket);
        if (!stream) {
            publish_error(Status::invalid_handle("Invalid StreamTicket"));
            return nullptr;
        }
        if (stream->owner != h) {
            publish_error(Status::invalid_handle("StreamTicket does not belong to this handle"));
            return nullptr;
        }

        auto core = stream->core;
        if (!core) {
            {
                std::lock_guard<std::mutex> hlock(h->task_mutex);
                h->stream_tickets.erase(stream);
            }
            LiveObjectRegistry::instance().unregister_object(stream);
            delete stream;
            publish_error(Status::invalid_handle("StreamTicket has null core"));
            return nullptr;
        }

        bool block_mode = core->request.output.stream_block;
        std::unique_lock<std::mutex> lock(core->mutex);

        if (block_mode) {
            core->cv.wait(lock, [&]() {
                return !core->chunks.empty() || core->is_terminal() || core->cancel_requested.load();
            });
        }

        if (!core->chunks.empty()) {
            core->current_chunk = std::move(core->chunks.front());
            core->chunks.pop_front();
            core->queue_not_full.notify_one();
            return core->current_chunk->result.text.c_str();
        }

        // Chunks queue is empty
        if (!core->is_terminal()) {
            // Task is still running, non-blocking mode: return empty string ""
            core->current_chunk.reset();
            return "";
        }

        // Chunks queue is empty AND task is terminal: this is EOF!
        core->current_chunk.reset();
        core->state = TaskState::Consumed;
        Status term_status = core->terminal_status;
        lock.unlock();

        {
            std::lock_guard<std::mutex> hlock(h->task_mutex);
            h->tasks.erase(core);
            h->stream_tickets.erase(stream);
        }
        LiveObjectRegistry::instance().unregister_object(stream);
        delete stream;

        if (!term_status.is_ok()) {
            publish_error(term_status);
        }
        return nullptr;
    } catch (const std::exception &e) {
        publish_error(Status::internal(e.what()));
        return nullptr;
    } catch (...) {
        publish_error(Status::internal());
        return nullptr;
    }
}

const int *inferlet_get_stream_tokens(void *task_stream, int *real_token_length) {
    clear_last_error();
    if (real_token_length) *real_token_length = 0;
    try {
        if (!task_stream) {
            publish_error(Status::invalid_handle("task_stream cannot be NULL"));
            return nullptr;
        }
        if (!real_token_length) {
            publish_error(Status::invalid_argument("real_token_length cannot be NULL"));
            return nullptr;
        }
        auto *stream = checked_cast<StreamTicketInternal>(task_stream, ObjectKind::StreamTicket);
        if (!stream || !stream->core) {
            publish_error(Status::invalid_handle("Invalid StreamTicket"));
            return nullptr;
        }
        auto core = stream->core;
        std::lock_guard<std::mutex> lock(core->mutex);
        if (!core->current_chunk.has_value() || core->current_chunk->result.tokens.empty()) {
            *real_token_length = 0;
            return nullptr;
        }
        *real_token_length = static_cast<int>(core->current_chunk->result.tokens.size());
        return core->current_chunk->result.tokens.data();
    } catch (const std::exception &e) {
        publish_error(Status::internal(e.what()));
        return nullptr;
    } catch (...) {
        publish_error(Status::internal());
        return nullptr;
    }
}

const int *inferlet_get_stream_token_offsets(void *task_stream) {
    clear_last_error();
    try {
        if (!task_stream) {
            publish_error(Status::invalid_handle("task_stream cannot be NULL"));
            return nullptr;
        }
        auto *stream = checked_cast<StreamTicketInternal>(task_stream, ObjectKind::StreamTicket);
        if (!stream || !stream->core) {
            publish_error(Status::invalid_handle("Invalid StreamTicket"));
            return nullptr;
        }
        auto core = stream->core;
        std::lock_guard<std::mutex> lock(core->mutex);
        if (!core->current_chunk.has_value() || core->current_chunk->result.byte_offsets.empty()) {
            return nullptr;
        }
        return core->current_chunk->result.byte_offsets.data();
    } catch (const std::exception &e) {
        publish_error(Status::internal(e.what()));
        return nullptr;
    } catch (...) {
        publish_error(Status::internal());
        return nullptr;
    }
}

const float *inferlet_get_stream_token_probability(void *task_stream) {
    clear_last_error();
    try {
        if (!task_stream) {
            publish_error(Status::invalid_handle("task_stream cannot be NULL"));
            return nullptr;
        }
        auto *stream = checked_cast<StreamTicketInternal>(task_stream, ObjectKind::StreamTicket);
        if (!stream || !stream->core) {
            publish_error(Status::invalid_handle("Invalid StreamTicket"));
            return nullptr;
        }
        auto core = stream->core;
        std::lock_guard<std::mutex> lock(core->mutex);
        if (!core->current_chunk.has_value() || core->current_chunk->result.probabilities.empty()) {
            return nullptr;
        }
        return core->current_chunk->result.probabilities.data();
    } catch (const std::exception &e) {
        publish_error(Status::internal(e.what()));
        return nullptr;
    } catch (...) {
        publish_error(Status::internal());
        return nullptr;
    }
}

const float **inferlet_get_stream_output_logits(void *task_stream, int *output_tokens_length, int *vocab_size) {
    clear_last_error();
    if (output_tokens_length) *output_tokens_length = 0;
    if (vocab_size) *vocab_size = 0;
    try {
        if (!task_stream) {
            publish_error(Status::invalid_handle("task_stream cannot be NULL"));
            return nullptr;
        }
        if (!output_tokens_length) {
            publish_error(Status::invalid_argument("output_tokens_length cannot be NULL"));
            return nullptr;
        }
        if (!vocab_size) {
            publish_error(Status::invalid_argument("vocab_size cannot be NULL"));
            return nullptr;
        }
        auto *stream = checked_cast<StreamTicketInternal>(task_stream, ObjectKind::StreamTicket);
        if (!stream || !stream->core) {
            publish_error(Status::invalid_handle("Invalid StreamTicket"));
            return nullptr;
        }
        auto core = stream->core;
        std::lock_guard<std::mutex> lock(core->mutex);
        if (!core->current_chunk.has_value() || core->current_chunk->result.logits.rows == 0) {
            *output_tokens_length = 0;
            *vocab_size = 0;
            return nullptr;
        }
        *output_tokens_length = core->current_chunk->result.logits.rows;
        *vocab_size = core->current_chunk->result.logits.cols;
        return core->current_chunk->result.logits.row_ptrs.data();
    } catch (const std::exception &e) {
        publish_error(Status::internal(e.what()));
        return nullptr;
    } catch (...) {
        publish_error(Status::internal());
        return nullptr;
    }
}

const float **inferlet_get_stream_output_embedding(void *task_stream, int *output_tokens_length, int *embedding_dim) {
    clear_last_error();
    if (output_tokens_length) *output_tokens_length = 0;
    if (embedding_dim) *embedding_dim = 0;
    try {
        if (!task_stream) {
            publish_error(Status::invalid_handle("task_stream cannot be NULL"));
            return nullptr;
        }
        if (!output_tokens_length) {
            publish_error(Status::invalid_argument("output_tokens_length cannot be NULL"));
            return nullptr;
        }
        if (!embedding_dim) {
            publish_error(Status::invalid_argument("embedding_dim cannot be NULL"));
            return nullptr;
        }
        auto *stream = checked_cast<StreamTicketInternal>(task_stream, ObjectKind::StreamTicket);
        if (!stream || !stream->core) {
            publish_error(Status::invalid_handle("Invalid StreamTicket"));
            return nullptr;
        }
        auto core = stream->core;
        std::lock_guard<std::mutex> lock(core->mutex);
        if (!core->current_chunk.has_value() || core->current_chunk->result.embeddings.rows == 0) {
            *output_tokens_length = 0;
            *embedding_dim = 0;
            return nullptr;
        }
        *output_tokens_length = core->current_chunk->result.embeddings.rows;
        *embedding_dim = core->current_chunk->result.embeddings.cols;
        return core->current_chunk->result.embeddings.row_ptrs.data();
    } catch (const std::exception &e) {
        publish_error(Status::internal(e.what()));
        return nullptr;
    } catch (...) {
        publish_error(Status::internal());
        return nullptr;
    }
}

const int *inferlet_get_stream_input_tokens(void *task_stream, int *input_tokens_length, const char ***input_tokens_text) {
    clear_last_error();
    if (input_tokens_length) *input_tokens_length = 0;
    if (input_tokens_text) *input_tokens_text = nullptr;
    try {
        if (!task_stream) {
            publish_error(Status::invalid_handle("task_stream cannot be NULL"));
            return nullptr;
        }
        if (!input_tokens_length) {
            publish_error(Status::invalid_argument("input_tokens_length cannot be NULL"));
            return nullptr;
        }
        if (!input_tokens_text) {
            publish_error(Status::invalid_argument("input_tokens_text cannot be NULL"));
            return nullptr;
        }
        auto *stream = checked_cast<StreamTicketInternal>(task_stream, ObjectKind::StreamTicket);
        if (!stream || !stream->core) {
            publish_error(Status::invalid_handle("Invalid StreamTicket"));
            return nullptr;
        }
        auto core = stream->core;
        std::lock_guard<std::mutex> lock(core->mutex);
        if (!core->current_chunk.has_value() || !core->current_chunk->input.has_value() ||
            core->current_chunk->input->tokens.empty()) {
            *input_tokens_length = 0;
            *input_tokens_text = nullptr;
            return nullptr;
        }
        *input_tokens_length = static_cast<int>(core->current_chunk->input->tokens.size());
        *input_tokens_text = core->current_chunk->input->text_ptrs.data();
        return core->current_chunk->input->tokens.data();
    } catch (const std::exception &e) {
        publish_error(Status::internal(e.what()));
        return nullptr;
    } catch (...) {
        publish_error(Status::internal());
        return nullptr;
    }
}

// =================== Parameter APIs ===================

void *inferlet_parameter_allocate(void) {
    clear_last_error();
    try {
        auto *param = new (std::nothrow) ParameterInternal();
        if (!param) {
            publish_error(Status::out_of_memory("Failed to allocate Parameter"));
            return nullptr;
        }
        LiveObjectRegistry::instance().register_object(param, ObjectKind::Parameter);
        return param;
    } catch (const std::bad_alloc &) {
        publish_error(Status::out_of_memory());
        return nullptr;
    } catch (const std::exception &e) {
        publish_error(Status::internal(e.what()));
        return nullptr;
    } catch (...) {
        publish_error(Status::internal());
        return nullptr;
    }
}

void inferlet_parameter_deallocate(void *param) {
    clear_last_error();
    if (!param) return;
    try {
        auto *p = checked_cast<ParameterInternal>(param, ObjectKind::Parameter);
        if (!p) {
            log_warn("inferlet_parameter_deallocate called with invalid handle");
            return;
        }
        LiveObjectRegistry::instance().unregister_object(p);
        delete p;
    } catch (const std::exception &e) {
        log_error(std::string("inferlet_parameter_deallocate exception: ") + e.what());
    } catch (...) {
        log_error("inferlet_parameter_deallocate unknown exception");
    }
}

void inferlet_parameter_set_tokenizer_only(void *param, int tokenizer_only) {
    clear_last_error();
    try {
        auto *p = checked_cast<ParameterInternal>(param, ObjectKind::Parameter);
        if (!p) {
            publish_error(Status::invalid_handle("Invalid Parameter handle"));
            return;
        }
        p->set_tokenizer_only(tokenizer_only);
    } catch (...) {
        publish_error(Status::internal());
    }
}

void inferlet_parameter_set_tokenizer_slow(void *param, int tokenizer_slow) {
    clear_last_error();
    try {
        auto *p = checked_cast<ParameterInternal>(param, ObjectKind::Parameter);
        if (!p) {
            publish_error(Status::invalid_handle("Invalid Parameter handle"));
            return;
        }
        p->set_tokenizer_slow(tokenizer_slow);
    } catch (...) {
        publish_error(Status::internal());
    }
}

void inferlet_parameter_set_vision_max_patch_cnt(void *param, int vision_max_patch_cnt) {
    clear_last_error();
    try {
        auto *p = checked_cast<ParameterInternal>(param, ObjectKind::Parameter);
        if (!p) {
            publish_error(Status::invalid_handle("Invalid Parameter handle"));
            return;
        }
        p->set_vision_max_patch_cnt(vision_max_patch_cnt);
    } catch (...) {
        publish_error(Status::internal());
    }
}

void inferlet_parameter_set_vision_max_grid_size(void *param, int vision_max_grid_size) {
    clear_last_error();
    try {
        auto *p = checked_cast<ParameterInternal>(param, ObjectKind::Parameter);
        if (!p) {
            publish_error(Status::invalid_handle("Invalid Parameter handle"));
            return;
        }
        p->set_vision_max_grid_size(vision_max_grid_size);
    } catch (...) {
        publish_error(Status::internal());
    }
}

void inferlet_parameter_set_vision_trans_skipped(void *param, int vision_transform_skipped) {
    clear_last_error();
    try {
        auto *p = checked_cast<ParameterInternal>(param, ObjectKind::Parameter);
        if (!p) {
            publish_error(Status::invalid_handle("Invalid Parameter handle"));
            return;
        }
        p->set_vision_trans_skipped(vision_transform_skipped);
    } catch (...) {
        publish_error(Status::internal());
    }
}

void inferlet_parameter_set_decoder_max_token_cnt(void *param, int decoder_max_token_count) {
    clear_last_error();
    try {
        auto *p = checked_cast<ParameterInternal>(param, ObjectKind::Parameter);
        if (!p) {
            publish_error(Status::invalid_handle("Invalid Parameter handle"));
            return;
        }
        p->set_decoder_max_token_cnt(decoder_max_token_count);
    } catch (...) {
        publish_error(Status::internal());
    }
}

void inferlet_parameter_set_device_id(void *param, int device_id) {
    clear_last_error();
    try {
        auto *p = checked_cast<ParameterInternal>(param, ObjectKind::Parameter);
        if (!p) {
            publish_error(Status::invalid_handle("Invalid Parameter handle"));
            return;
        }
        p->set_device_id(device_id);
    } catch (...) {
        publish_error(Status::internal());
    }
}

void inferlet_parameter_set_device_list(void *param, int *device_list, int device_cnt) {
    clear_last_error();
    try {
        auto *p = checked_cast<ParameterInternal>(param, ObjectKind::Parameter);
        if (!p) {
            publish_error(Status::invalid_handle("Invalid Parameter handle"));
            return;
        }
        p->set_device_list(device_list, device_cnt);
    } catch (...) {
        publish_error(Status::internal());
    }
}

void inferlet_parameter_set_backend_cnt(void *param, int backend_cnt) {
    clear_last_error();
    try {
        auto *p = checked_cast<ParameterInternal>(param, ObjectKind::Parameter);
        if (!p) {
            publish_error(Status::invalid_handle("Invalid Parameter handle"));
            return;
        }
        p->set_backend_cnt(backend_cnt);
    } catch (...) {
        publish_error(Status::internal());
    }
}

void inferlet_parameter_set_low_rank_adapter(void *param, int lora_adapter_id, const char *lora_adapter_file) {
    clear_last_error();
    try {
        auto *p = checked_cast<ParameterInternal>(param, ObjectKind::Parameter);
        if (!p) {
            publish_error(Status::invalid_handle("Invalid Parameter handle"));
            return;
        }
        p->set_low_rank_adapter(lora_adapter_id, lora_adapter_file);
    } catch (...) {
        publish_error(Status::internal());
    }
}

void inferlet_parameter_set_load_from_file_sync(void *param, int load_sync) {
    clear_last_error();
    try {
        auto *p = checked_cast<ParameterInternal>(param, ObjectKind::Parameter);
        if (!p) {
            publish_error(Status::invalid_handle("Invalid Parameter handle"));
            return;
        }
        p->set_load_from_file_sync(load_sync);
    } catch (...) {
        publish_error(Status::internal());
    }
}

void inferlet_parameter_set_run_config_file(void *param, const char *run_config_file) {
    clear_last_error();
    try {
        auto *p = checked_cast<ParameterInternal>(param, ObjectKind::Parameter);
        if (!p) {
            publish_error(Status::invalid_handle("Invalid Parameter handle"));
            return;
        }
        p->set_run_config_file(run_config_file);
    } catch (...) {
        publish_error(Status::internal());
    }
}

// =================== Tokenizer APIs ===================

int inferlet_tokenizer_encode(void *handle, const char *text, int text_length, int *tokens, int max_token_length, int *real_token_length, int add_special_token, int parse_special_token) {
    clear_last_error();
    if (real_token_length) *real_token_length = 0;
    try {
        if (!handle) return publish_error(Status::invalid_handle("handle cannot be NULL"));
        if (!text) return publish_error(Status::invalid_argument("text cannot be NULL"));
        if (text_length < -1) return publish_error(Status::invalid_argument("text_length cannot be less than -1"));
        if (!real_token_length) return publish_error(Status::invalid_argument("real_token_length cannot be NULL"));
        if (max_token_length < 0) return publish_error(Status::invalid_argument("max_token_length cannot be negative"));
        if (max_token_length > 0 && !tokens) return publish_error(Status::invalid_argument("tokens cannot be NULL when max_token_length > 0"));

        auto *h = checked_cast<ModelHandleInternal>(handle, ObjectKind::ModelHandle);
        if (!h) return publish_error(Status::invalid_handle("Invalid ModelHandle"));

        Status ready_st = h->wait_for_ready();
        if (!ready_st) return publish_error(ready_st);

        const llama_vocab *vocab = h->vocab();
        if (!vocab) return publish_error(Status::invalid_handle("Model vocab is NULL"));

        size_t actual_len = (text_length == -1) ? std::strlen(text) : static_cast<size_t>(text_length);
        std::string_view sv(text, actual_len);
        if (!is_valid_utf8(sv)) return publish_error(Status::invalid_argument("Invalid UTF-8 in text"));

        Status st = TokenCodec::tokenize(vocab, sv, tokens, max_token_length, *real_token_length,
                                         add_special_token != 0, parse_special_token != 0);
        return publish_status(st);
    } catch (const std::bad_alloc &) {
        return publish_error(Status::out_of_memory());
    } catch (const std::exception &e) {
        return publish_error(Status::internal(e.what()));
    } catch (...) {
        return publish_error(Status::internal());
    }
}

int inferlet_tokenizer_decode(void *handle, const int *tokens, int token_length, char *text, int max_text_length, int *real_text_length, int skip_special_tokens) {
    clear_last_error();
    if (real_text_length) *real_text_length = 0;
    try {
        if (!handle) return publish_error(Status::invalid_handle("handle cannot be NULL"));
        if (token_length < 0) return publish_error(Status::invalid_argument("token_length cannot be negative"));
        if (token_length > 0 && !tokens) return publish_error(Status::invalid_argument("tokens cannot be NULL when token_length > 0"));
        if (!real_text_length) return publish_error(Status::invalid_argument("real_text_length cannot be NULL"));
        if (max_text_length < 0) return publish_error(Status::invalid_argument("max_text_length cannot be negative"));
        if (max_text_length > 0 && !text) return publish_error(Status::invalid_argument("text cannot be NULL when max_text_length > 0"));

        auto *h = checked_cast<ModelHandleInternal>(handle, ObjectKind::ModelHandle);
        if (!h) return publish_error(Status::invalid_handle("Invalid ModelHandle"));

        Status ready_st = h->wait_for_ready();
        if (!ready_st) return publish_error(ready_st);

        const llama_vocab *vocab = h->vocab();
        if (!vocab) return publish_error(Status::invalid_handle("Model vocab is NULL"));

        Status st = TokenCodec::detokenize(vocab, tokens, token_length, text, max_text_length,
                                           *real_text_length, skip_special_tokens != 0);
        return publish_status(st);
    } catch (const std::bad_alloc &) {
        return publish_error(Status::out_of_memory());
    } catch (const std::exception &e) {
        return publish_error(Status::internal(e.what()));
    } catch (...) {
        return publish_error(Status::internal());
    }
}

} // extern "C"

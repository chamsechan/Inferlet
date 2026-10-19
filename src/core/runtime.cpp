#include "runtime.h"
#include "engine/llama_engine.h"
#include "support/logging.h"

namespace inferlet {

Runtime &Runtime::instance() {
    static Runtime instance;
    return instance;
}

Runtime::~Runtime() {
    std::lock_guard<std::mutex> lock(mutex_);
    init_ref_count_ = 0;
    live_handle_count_ = 0;
    try_shutdown_locked();
}

Status Runtime::init() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (init_ref_count_ == 0) {
        if (!backend_initialized_) {
            log_debug("Initializing llama backend and scheduler");
            global_llama_backend_init();
            backend_initialized_ = true;
        }
        scheduler_.start();
    }
    init_ref_count_++;
    return Status::ok();
}

void Runtime::deinit() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (init_ref_count_ == 0) {
        log_warn("inferlet_deinit called without matching Init");
        return;
    }
    init_ref_count_--;
    if (init_ref_count_ == 0 && live_handle_count_ == 0) {
        try_shutdown_locked();
    } else if (init_ref_count_ == 0) {
        log_info("inferlet_deinit: deferred shutdown waiting for active handles");
    }
}

bool Runtime::is_initialized() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return init_ref_count_ > 0;
}

Status Runtime::register_handle() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (init_ref_count_ == 0) {
        return Status::invalid_state("Inferlet is not initialized; call inferlet_init first");
    }
    live_handle_count_++;
    return Status::ok();
}

void Runtime::unregister_handle() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (live_handle_count_ > 0) {
        live_handle_count_--;
    }
    if (init_ref_count_ == 0 && live_handle_count_ == 0) {
        try_shutdown_locked();
    }
}

void Runtime::try_shutdown_locked() {
    if (!backend_initialized_) {
        return;
    }
    log_debug("Shutting down scheduler and llama backend");
    scheduler_.stop();
    global_llama_backend_free();
    backend_initialized_ = false;
}

uint32_t Runtime::init_ref_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return init_ref_count_;
}

uint32_t Runtime::live_handle_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return live_handle_count_;
}

bool Runtime::backend_initialized() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return backend_initialized_;
}

} // namespace inferlet

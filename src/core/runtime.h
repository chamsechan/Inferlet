#ifndef INFERLET_CORE_RUNTIME_H_INCLUDED
#define INFERLET_CORE_RUNTIME_H_INCLUDED

#include "support/status.h"
#include "scheduler.h"
#include <mutex>
#include <cstdint>

namespace inferlet {

class Runtime {
public:
    static Runtime &instance();

    Status init();
    void deinit();

    bool is_initialized() const;
    Status register_handle();
    void unregister_handle();

    uint32_t init_ref_count() const;
    uint32_t live_handle_count() const;
    bool backend_initialized() const;

    Scheduler &scheduler() { return scheduler_; }

private:
    Runtime() = default;
    ~Runtime();
    Runtime(const Runtime &) = delete;
    Runtime &operator=(const Runtime &) = delete;

    void try_shutdown_locked();

    mutable std::mutex mutex_;
    uint32_t init_ref_count_ = 0;
    uint32_t live_handle_count_ = 0;
    bool backend_initialized_ = false;
    Scheduler scheduler_;
};

} // namespace inferlet

#endif // INFERLET_CORE_RUNTIME_H_INCLUDED

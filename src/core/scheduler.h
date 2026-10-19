#ifndef INFERLET_CORE_SCHEDULER_H_INCLUDED
#define INFERLET_CORE_SCHEDULER_H_INCLUDED

#include "support/status.h"
#include <atomic>
#include <cstddef>
#include <mutex>
#include <vector>
#include <deque>
#include <thread>
#include <condition_variable>
#include <memory>

namespace inferlet {

struct TaskCore;

class Scheduler {
public:
    Scheduler() = default;
    ~Scheduler();

    void start(size_t worker_count = 0);
    void stop();
    bool is_running() const;

    Status submit(std::shared_ptr<TaskCore> task);

private:
    void worker_loop();

    std::atomic<bool> running_{false};
    std::vector<std::thread> workers_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<std::shared_ptr<TaskCore>> queue_;
};

} // namespace inferlet

#endif // INFERLET_CORE_SCHEDULER_H_INCLUDED

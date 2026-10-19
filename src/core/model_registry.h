#ifndef INFERLET_CORE_MODEL_REGISTRY_H_INCLUDED
#define INFERLET_CORE_MODEL_REGISTRY_H_INCLUDED

#include "engine/llama_engine.h"
#include "model_config.h"
#include "support/status.h"
#include <string>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <unordered_map>
#include <atomic>
#include <thread>

namespace inferlet {

class LoraRegistry;
class MtmdBackend;

enum class LoadState {
    Loading,
    Ready,
    Failed,
    Closing
};

class Semaphore {
public:
    explicit Semaphore(int count = 1) : count_(count) {}

    void acquire() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this]() { return count_ > 0; });
        --count_;
    }

    bool try_acquire() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (count_ > 0) {
            --count_;
            return true;
        }
        return false;
    }

    template <typename Rep, typename Period>
    bool try_acquire_for(const std::chrono::duration<Rep, Period> &timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!cv_.wait_for(lock, timeout, [this]() { return count_ > 0; })) {
            return false;
        }
        --count_;
        return true;
    }

    void release() {
        std::lock_guard<std::mutex> lock(mutex_);
        ++count_;
        cv_.notify_one();
    }

    int count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return count_;
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    int count_;
};

struct SharedModel {
    std::string canonical_path;
    std::string fingerprint;
    ModelConfig config;

    LlamaModelPtr model;
    const llama_vocab *vocab = nullptr;
    std::shared_ptr<LoraRegistry> lora_registry;
    std::shared_ptr<MtmdBackend> multimodal;

    mutable std::mutex mutex;
    std::condition_variable load_cv;
    LoadState load_state = LoadState::Loading;
    Status load_status;

    std::atomic<int> ref_count{0};
    std::atomic<int> queued{0};
    std::atomic<int> running{0};
    std::shared_ptr<Semaphore> exec_semaphore;

    std::thread loader_thread;

    ~SharedModel() {
        if (loader_thread.joinable() && loader_thread.get_id() != std::this_thread::get_id()) {
            loader_thread.join();
        }
    }

    bool is_ready() const {
        std::lock_guard<std::mutex> lock(mutex);
        return load_state == LoadState::Ready;
    }
};

class ModelRegistry {
public:
    static ModelRegistry &instance();

    Status get_or_load(const std::string &path,
                       const ModelConfig &config,
                       std::shared_ptr<SharedModel> &out_model);

    void release(const std::shared_ptr<SharedModel> &model);
    void clear();

private:
    ModelRegistry() = default;
    ~ModelRegistry() = default;
    ModelRegistry(const ModelRegistry &) = delete;
    ModelRegistry &operator=(const ModelRegistry &) = delete;

    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::shared_ptr<SharedModel>> models_;
};

} // namespace inferlet

#endif // INFERLET_CORE_MODEL_REGISTRY_H_INCLUDED

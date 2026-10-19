#include "model_registry.h"
#include "engine/lora_registry.h"
#include "multimodal/mtmd_backend.h"
#include "support/logging.h"
#include <filesystem>

namespace inferlet {

ModelRegistry &ModelRegistry::instance() {
    static ModelRegistry instance;
    return instance;
}

Status ModelRegistry::get_or_load(const std::string &path,
                                  const ModelConfig &config,
                                  std::shared_ptr<SharedModel> &out_model) {
    std::error_code ec;
    std::string canon_path = std::filesystem::weakly_canonical(path, ec).string();
    if (ec || canon_path.empty()) {
        canon_path = std::filesystem::path(path).lexically_normal().string();
    }

    std::string fp = config.compute_fingerprint();

    std::unique_lock<std::mutex> lock(mutex_);
    auto it = models_.find(canon_path);
    if (it != models_.end()) {
        auto existing = it->second;
        if (existing->fingerprint != fp) {
            return Status::invalid_argument("Model already loaded with different parameters: " + canon_path);
        }
        existing->ref_count++;
        lock.unlock();

        if (config.load_sync) {
            std::unique_lock<std::mutex> lk(existing->mutex);
            existing->load_cv.wait(lk, [&existing]() {
                return existing->load_state != LoadState::Loading;
            });
            if (existing->load_state == LoadState::Failed) {
                Status fail_st = existing->load_status;
                lk.unlock();
                release(existing);
                return fail_st.is_ok() ? Status::model_load() : fail_st;
            }
        }
        out_model = existing;
        log_info("Reusing existing shared model: " + canon_path);
        return Status::ok();
    }

    // New load
    auto shared = std::make_shared<SharedModel>();
    shared->canonical_path = canon_path;
    shared->fingerprint = fp;
    shared->config = config;
    shared->load_state = LoadState::Loading;
    shared->ref_count = 1;
    shared->exec_semaphore = std::make_shared<Semaphore>(config.backend_count > 0 ? config.backend_count : 1);

    models_[canon_path] = shared;

    // Start background load
    shared->loader_thread = std::thread([shared, config, canon_path]() {
        log_info("Starting async load for: " + canon_path);
        Status load_st = LlamaEngine::load_model(config, canon_path, shared->model, shared->vocab);
        if (load_st && !config.lora_files.empty() && shared->model) {
            shared->lora_registry = std::make_shared<LoraRegistry>();
            load_st = shared->lora_registry->load_adapters(shared->model.get(), config.lora_files);
        }
        if (load_st && !config.parsed_run_config.vision.mmproj.empty()) {
            load_st = MtmdBackend::load(config, shared->model.get(), shared->multimodal);
        }
        {
            std::lock_guard<std::mutex> lk(shared->mutex);
            if (load_st) {
                shared->load_state = LoadState::Ready;
                shared->load_status = Status::ok();
                log_info("Async load completed successfully for: " + canon_path);
            } else {
                shared->load_state = LoadState::Failed;
                shared->load_status = Status::model_load(load_st.message.empty() ? "Model load failed" : load_st.message);
                log_error("Async load failed for: " + canon_path + ": " + shared->load_status.message);
            }
        }
        shared->load_cv.notify_all();
    });

    lock.unlock();

    if (config.load_sync) {
        std::unique_lock<std::mutex> lk(shared->mutex);
        shared->load_cv.wait(lk, [&shared]() {
            return shared->load_state != LoadState::Loading;
        });
        if (shared->load_state == LoadState::Failed) {
            Status fail_st = shared->load_status;
            lk.unlock();
            release(shared);
            return fail_st.is_ok() ? Status::model_load() : fail_st;
        }
    }

    out_model = shared;
    log_info("Registered new shared model (loading): " + canon_path);
    return Status::ok();
}

void ModelRegistry::release(const std::shared_ptr<SharedModel> &model) {
    if (!model) return;
    std::shared_ptr<SharedModel> to_release;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        model->ref_count--;
        if (model->ref_count <= 0) {
            log_info("All references dropped, releasing shared model: " + model->canonical_path);
            models_.erase(model->canonical_path);
            to_release = model;
        }
    }
    if (to_release) {
        if (to_release->loader_thread.joinable() && to_release->loader_thread.get_id() != std::this_thread::get_id()) {
            to_release->loader_thread.join();
        }
    }
}

void ModelRegistry::clear() {
    std::unordered_map<std::string, std::shared_ptr<SharedModel>> to_clear;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        to_clear = std::move(models_);
    }
    for (auto &kv : to_clear) {
        if (kv.second && kv.second->loader_thread.joinable() && kv.second->loader_thread.get_id() != std::this_thread::get_id()) {
            kv.second->loader_thread.join();
        }
    }
}

} // namespace inferlet

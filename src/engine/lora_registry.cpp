#include "lora_registry.h"
#include "support/logging.h"

namespace inferlet {

Status LoraRegistry::load_adapters(llama_model *model, const std::map<int, std::string> &lora_files) {
    if (!model) {
        return Status::invalid_handle("Base model is NULL when loading LoRA adapters");
    }

    std::lock_guard<std::mutex> lock(mutex_);
    adapters_.clear();

    for (const auto &[id, path] : lora_files) {
        if (id <= 0) {
            return Status::invalid_argument("LoRA adapter ID must be > 0");
        }
        if (path.empty()) {
            return Status::invalid_argument("LoRA adapter file path cannot be empty");
        }
        log_info("Loading LoRA adapter id=" + std::to_string(id) + " from: " + path);
        LlamaLoraPtr lora_ptr;
        Status st = LlamaEngine::load_lora(model, path, lora_ptr);
        if (!st) {
            log_error("Failed to load LoRA adapter id=" + std::to_string(id) + " from: " + path + ": " + st.message);
            adapters_.clear();
            return st;
        }
        adapters_[id] = std::move(lora_ptr);
    }
    return Status::ok();
}

llama_adapter_lora *LoraRegistry::get_adapter(int lora_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = adapters_.find(lora_id);
    if (it != adapters_.end()) {
        return it->second.get();
    }
    return nullptr;
}

bool LoraRegistry::has_adapter(int lora_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return adapters_.find(lora_id) != adapters_.end();
}

size_t LoraRegistry::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return adapters_.size();
}

void LoraRegistry::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    adapters_.clear();
}

} // namespace inferlet

#ifndef INFERLET_ENGINE_LORA_REGISTRY_H_INCLUDED
#define INFERLET_ENGINE_LORA_REGISTRY_H_INCLUDED

#include "engine/llama_engine.h"
#include "support/status.h"
#include <map>
#include <unordered_map>
#include <string>
#include <mutex>
#include <memory>

namespace inferlet {

class LoraRegistry {
public:
    LoraRegistry() = default;
    ~LoraRegistry() = default;

    // Load all LoRAs specified in ModelConfig
    Status load_adapters(llama_model *model, const std::map<int, std::string> &lora_files);

    // Retrieve adapter by ID (thread-safe, read-only lookup)
    llama_adapter_lora *get_adapter(int lora_id) const;

    // Check if ID is registered
    bool has_adapter(int lora_id) const;

    // Return count of loaded adapters
    size_t size() const;

    // Clear adapters
    void clear();

private:
    mutable std::mutex mutex_;
    std::unordered_map<int, LlamaLoraPtr> adapters_;
};

} // namespace inferlet

#endif // INFERLET_ENGINE_LORA_REGISTRY_H_INCLUDED

#include "model_config.h"
#include <sstream>
#include <unordered_set>
#include <cstring>

namespace inferlet {

static bool is_power_of_two(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

void ModelConfig::mark_invalid(std::string reason) {
    valid = false;
    if (!invalid_reason.empty()) {
        invalid_reason += "; ";
    }
    invalid_reason += std::move(reason);
}

std::string ModelConfig::compute_fingerprint() const {
    std::ostringstream oss;
    oss << "tok_only=" << (tokenizer_only ? 1 : 0) << ";";
    oss << "tok_slow=" << (tokenizer_slow ? 1 : 0) << ";";
    oss << "vision_patch=" << vision_max_patch_cnt << ";";
    oss << "vision_grid=" << vision_max_grid_size << ";";
    oss << "vision_skip=" << (vision_transform_skipped ? 1 : 0) << ";";
    oss << "decoder_batch=" << decoder_max_token_count << ";";
    oss << "backends=" << backend_count << ";";
    oss << "dev_mode=" << static_cast<int>(devices.mode) << ";";
    if (devices.mode == DeviceMode::Single) {
        oss << "single_dev=" << devices.single_device_id << ";";
    } else if (devices.mode == DeviceMode::List) {
        oss << "devs=[";
        for (size_t i = 0; i < devices.device_list.size(); ++i) {
            if (i > 0) oss << ",";
            oss << devices.device_list[i];
        }
        oss << "];";
    }
    oss << "loras=[";
    for (const auto &[id, path] : lora_files) {
        oss << id << ":" << path << ",";
    }
    oss << "];";
    const auto &run = parsed_run_config;
    oss << "mmproj=" << run.vision.mmproj.size() << ":" << run.vision.mmproj << ";";
    oss << "ctx=" << run.model.context_size << ";";
    oss << "ubatch=" << run.model.n_ubatch << ";";
    oss << "threads=" << run.model.threads << ";";
    oss << "threads_batch=" << run.model.threads_batch << ";";
    oss << "mmap=" << (run.model.mmap ? 1 : 0) << ";";
    oss << "mlock=" << (run.model.mlock ? 1 : 0) << ";";
    oss << "gpu_layers=" << run.model.gpu_layers << ";";
    oss << "queue=" << run.scheduler.queue_capacity << ";";
    oss << "stream_queue=" << run.scheduler.stream_queue_chunks << ";";
    oss << "seed=" << run.sampling.seed << ";";
    oss << "repeat=" << run.sampling.repeat_last_n << ";";
    oss << "beam_penalty=" << run.sampling.beam_length_penalty << ";";
    oss << "input_limit=" << run.limits.max_input_bytes << ";";
    oss << "multimodal_limit=" << run.limits.max_multimodal_bytes << ";";
    oss << "verbose_limit=" << run.limits.max_verbose_bytes << ";";
    oss << "visual_cache_limit=" << run.limits.max_visual_cache_bytes << ";";
    oss << "log=" << run.logging.level << ";";
    return oss.str();
}

Status ModelConfig::load_and_merge_run_config() {
    if (run_config_file.empty()) {
        return Status::ok();
    }
    RunConfig loaded;
    Status st = load_run_config_file(run_config_file, loaded);
    if (!st) {
        return st;
    }
    parsed_run_config = loaded;

    // Apply JSON config to ModelConfig fields if NOT explicitly set via setter
    if (explicitly_set.find(ConfigField::DecoderMaxTokenCount) == explicitly_set.end()) {
        decoder_max_token_count = loaded.model.n_batch;
    }
    return Status::ok();
}

Status ModelConfig::validate_supported() const {
    if (tokenizer_slow) {
        return Status::unsupported("Slow-tokenizer mode is not supported by the llama.cpp backend");
    }
    if (vision_max_patch_cnt != 0 || vision_max_grid_size != 0) {
        return Status::unsupported("Custom vision patch/grid limits are unsupported; libmtmd uses the projector's preprocessing rules");
    }
    if (vision_transform_skipped) {
        return Status::unsupported("Precomputed visual embeddings are not supported; provide raw RGB images");
    }
    if (tokenizer_only && !parsed_run_config.vision.mmproj.empty()) {
        return Status::unsupported("Visual projector cannot be loaded in tokenizer-only mode");
    }
    if (parsed_run_config.scheduler.worker_count != 0) {
        return Status::unsupported("Per-model scheduler worker_count is not supported");
    }
    return Status::ok();
}

void ParameterInternal::set_tokenizer_only(int val) {
    if (val != 0 && val != 1) {
        config.mark_invalid("tokenizer_only must be 0 or 1");
        return;
    }
    config.tokenizer_only = (val != 0);
    config.explicitly_set.insert(ConfigField::TokenizerOnly);
}

void ParameterInternal::set_tokenizer_slow(int val) {
    if (val != 0 && val != 1) {
        config.mark_invalid("tokenizer_slow must be 0 or 1");
        return;
    }
    config.tokenizer_slow = (val != 0);
    config.explicitly_set.insert(ConfigField::TokenizerSlow);
}

void ParameterInternal::set_vision_max_patch_cnt(int val) {
    if (val < 0) {
        config.mark_invalid("vision_max_patch_cnt must be >= 0");
        return;
    }
    config.vision_max_patch_cnt = val;
    config.explicitly_set.insert(ConfigField::VisionMaxPatchCnt);
}

void ParameterInternal::set_vision_max_grid_size(int val) {
    if (val < 0) {
        config.mark_invalid("vision_max_grid_size must be >= 0");
        return;
    }
    config.vision_max_grid_size = val;
    config.explicitly_set.insert(ConfigField::VisionMaxGridSize);
}

void ParameterInternal::set_vision_trans_skipped(int val) {
    if (val != 0 && val != 1) {
        config.mark_invalid("vision_transform_skipped must be 0 or 1");
        return;
    }
    config.vision_transform_skipped = (val != 0);
    config.explicitly_set.insert(ConfigField::VisionTransformSkipped);
}

void ParameterInternal::set_decoder_max_token_cnt(int val) {
    if (val <= 0) {
        config.mark_invalid("decoder_max_token_count must be > 0");
        return;
    }
    config.decoder_max_token_count = val;
    config.explicitly_set.insert(ConfigField::DecoderMaxTokenCount);
}

void ParameterInternal::set_device_id(int val) {
    if (val < -1) {
        config.mark_invalid("device_id must be >= -1");
        return;
    }
    config.devices.mode = (val == -1) ? DeviceMode::Auto : DeviceMode::Single;
    config.devices.single_device_id = val;
    config.devices.device_list.clear();
    config.explicitly_set.insert(ConfigField::DeviceId);
    config.explicitly_set.erase(ConfigField::DeviceList);
}

void ParameterInternal::set_device_list(const int *device_list, int device_cnt) {
    if (device_cnt < 0) {
        config.mark_invalid("device_cnt cannot be negative");
        return;
    }
    if (device_cnt == 0) {
        config.devices.mode = DeviceMode::Auto;
        config.devices.single_device_id = -1;
        config.devices.device_list.clear();
        config.explicitly_set.insert(ConfigField::DeviceList);
        config.explicitly_set.erase(ConfigField::DeviceId);
        return;
    }

    config.devices.mode = DeviceMode::List;
    config.devices.device_list.clear();
    config.devices.single_device_id = -1;

    if (!device_list) {
        // According to header contract: if device_list is NULL, auto-generate [0, 1, ..., device_cnt - 1]
        for (int i = 0; i < device_cnt; ++i) {
            config.devices.device_list.push_back(i);
        }
    } else {
        std::unordered_set<int> seen;
        for (int i = 0; i < device_cnt; ++i) {
            int dev = device_list[i];
            if (dev < 0) {
                config.mark_invalid("device_id in device_list must be >= 0");
                return;
            }
            if (!seen.insert(dev).second) {
                config.mark_invalid("duplicate device_id in device_list: " + std::to_string(dev));
                return;
            }
            config.devices.device_list.push_back(dev);
        }
    }
    config.explicitly_set.insert(ConfigField::DeviceList);
    config.explicitly_set.erase(ConfigField::DeviceId);

    // If backend_count was already set, re-validate relation
    if (config.explicitly_set.find(ConfigField::BackendCount) != config.explicitly_set.end()) {
        int dev_cnt = static_cast<int>(config.devices.device_list.size());
        if (dev_cnt % config.backend_count != 0) {
            config.mark_invalid("device_cnt must be divisible by backend_cnt");
        } else if (!is_power_of_two(dev_cnt / config.backend_count)) {
            config.mark_invalid("device_cnt / backend_cnt must be a power of 2");
        }
    }
}

void ParameterInternal::set_backend_cnt(int val) {
    if (val <= 0) {
        config.mark_invalid("backend_cnt must be > 0");
        return;
    }
    config.backend_count = val;
    config.explicitly_set.insert(ConfigField::BackendCount);

    if (config.devices.mode == DeviceMode::List && !config.devices.device_list.empty()) {
        int dev_cnt = static_cast<int>(config.devices.device_list.size());
        if (dev_cnt % val != 0) {
            config.mark_invalid("device_cnt must be divisible by backend_cnt");
        } else if (!is_power_of_two(dev_cnt / val)) {
            config.mark_invalid("device_cnt / backend_cnt must be a power of 2");
        }
    }
}

void ParameterInternal::set_low_rank_adapter(int id, const char *file) {
    if (id <= 0) {
        config.mark_invalid("lora_adapter_id must be > 0");
        return;
    }
    if (!file || std::strlen(file) == 0) {
        config.mark_invalid("lora_adapter_file cannot be empty");
        return;
    }
    config.lora_files[id] = std::string(file);
    config.explicitly_set.insert(ConfigField::LowRankAdapter);
}

void ParameterInternal::set_load_from_file_sync(int val) {
    if (val != 0 && val != 1) {
        config.mark_invalid("load_sync must be 0 or 1");
        return;
    }
    config.load_sync = (val != 0);
    config.explicitly_set.insert(ConfigField::LoadFromFileSync);
}

void ParameterInternal::set_run_config_file(const char *file) {
    if (!file) {
        config.mark_invalid("run_config_file cannot be NULL");
        return;
    }
    config.run_config_file = std::string(file);
    config.explicitly_set.insert(ConfigField::RunConfigFile);
}

} // namespace inferlet

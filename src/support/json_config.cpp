#include "json_config.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <sstream>
#include <set>
#include <filesystem>

namespace inferlet {

Status parse_run_config_json(std::string_view json_content, RunConfig &out_config) {
    nlohmann::json j;
    try {
        j = nlohmann::json::parse(json_content);
    } catch (const std::exception &e) {
        return Status::invalid_argument(std::string("Invalid JSON syntax: ") + e.what());
    }

    if (!j.is_object()) {
        return Status::invalid_argument("Run config root must be a JSON object");
    }

    // Check unknown top-level keys
    static const std::set<std::string> allowed_top_keys = {
        "schema_version", "model", "vision", "scheduler", "sampling", "limits", "logging"
    };
    for (auto it = j.begin(); it != j.end(); ++it) {
        if (allowed_top_keys.find(it.key()) == allowed_top_keys.end()) {
            return Status::invalid_argument("Unknown top-level key in run config: " + it.key());
        }
    }

    if (!j.contains("schema_version")) {
        return Status::invalid_argument("Run config missing required field 'schema_version'");
    }
    if (!j["schema_version"].is_number_integer()) {
        return Status::invalid_argument("'schema_version' must be an integer");
    }
    int version = j["schema_version"].get<int>();
    if (version != 1) {
        return Status::invalid_argument("Unsupported schema_version: " + std::to_string(version));
    }
    out_config.schema_version = version;

    // Parse model
    if (j.contains("model")) {
        const auto &m = j["model"];
        if (!m.is_object()) return Status::invalid_argument("'model' must be an object");

        if (m.contains("context_size")) {
            if (!m["context_size"].is_number_integer() || m["context_size"].get<int>() <= 0) {
                return Status::invalid_argument("'context_size' must be a positive integer");
            }
            out_config.model.context_size = m["context_size"].get<int>();
        }
        if (m.contains("n_batch")) {
            if (!m["n_batch"].is_number_integer() || m["n_batch"].get<int>() <= 0) {
                return Status::invalid_argument("'n_batch' must be a positive integer");
            }
            out_config.model.n_batch = m["n_batch"].get<int>();
        }
        if (m.contains("n_ubatch")) {
            if (!m["n_ubatch"].is_number_integer() || m["n_ubatch"].get<int>() <= 0) {
                return Status::invalid_argument("'n_ubatch' must be a positive integer");
            }
            out_config.model.n_ubatch = m["n_ubatch"].get<int>();
        }
        if (m.contains("threads")) {
            if (!m["threads"].is_number_integer() || m["threads"].get<int>() < 0) {
                return Status::invalid_argument("'threads' must be a non-negative integer");
            }
            out_config.model.threads = m["threads"].get<int>();
        }
        if (m.contains("threads_batch")) {
            if (!m["threads_batch"].is_number_integer() || m["threads_batch"].get<int>() < 0) {
                return Status::invalid_argument("'threads_batch' must be a non-negative integer");
            }
            out_config.model.threads_batch = m["threads_batch"].get<int>();
        }
        if (m.contains("mmap")) {
            if (!m["mmap"].is_boolean()) return Status::invalid_argument("'mmap' must be a boolean");
            out_config.model.mmap = m["mmap"].get<bool>();
        }
        if (m.contains("mlock")) {
            if (!m["mlock"].is_boolean()) return Status::invalid_argument("'mlock' must be a boolean");
            out_config.model.mlock = m["mlock"].get<bool>();
        }
        if (m.contains("gpu_layers")) {
            if (!m["gpu_layers"].is_number_integer() || m["gpu_layers"].get<int>() < 0) {
                return Status::invalid_argument("'gpu_layers' must be a non-negative integer");
            }
            out_config.model.gpu_layers = m["gpu_layers"].get<int>();
        }
    }

    if (j.contains("vision")) {
        const auto &v = j["vision"];
        if (!v.is_object()) return Status::invalid_argument("'vision' must be an object");
        for (auto it = v.begin(); it != v.end(); ++it) {
            if (it.key() != "mmproj") return Status::invalid_argument("Unknown vision key: " + it.key());
        }
        if (!v.contains("mmproj") || !v["mmproj"].is_string() || v["mmproj"].get<std::string>().empty()) {
            return Status::invalid_argument("'vision.mmproj' must be a non-empty file path");
        }
        out_config.vision.mmproj = v["mmproj"].get<std::string>();
        if (out_config.vision.mmproj.find('\0') != std::string::npos) {
            return Status::invalid_argument("'vision.mmproj' cannot contain NUL");
        }
    }

    // Parse scheduler
    if (j.contains("scheduler")) {
        const auto &s = j["scheduler"];
        if (!s.is_object()) return Status::invalid_argument("'scheduler' must be an object");

        if (s.contains("worker_count")) {
            if (!s["worker_count"].is_number_integer() || s["worker_count"].get<int>() < 0) {
                return Status::invalid_argument("'worker_count' must be a non-negative integer");
            }
            out_config.scheduler.worker_count = s["worker_count"].get<int>();
        }
        if (s.contains("queue_capacity")) {
            if (!s["queue_capacity"].is_number_integer() || s["queue_capacity"].get<int>() <= 0) {
                return Status::invalid_argument("'queue_capacity' must be a positive integer");
            }
            out_config.scheduler.queue_capacity = s["queue_capacity"].get<int>();
        }
        if (s.contains("stream_queue_chunks")) {
            if (!s["stream_queue_chunks"].is_number_integer() || s["stream_queue_chunks"].get<int>() <= 0) {
                return Status::invalid_argument("'stream_queue_chunks' must be a positive integer");
            }
            out_config.scheduler.stream_queue_chunks = s["stream_queue_chunks"].get<int>();
        }
    }

    // Parse sampling
    if (j.contains("sampling")) {
        const auto &s = j["sampling"];
        if (!s.is_object()) return Status::invalid_argument("'sampling' must be an object");

        if (s.contains("seed")) {
            if (!s["seed"].is_number_unsigned()) return Status::invalid_argument("'seed' must be an unsigned integer");
            out_config.sampling.seed = s["seed"].get<uint64_t>();
        }
        if (s.contains("repeat_last_n")) {
            if (!s["repeat_last_n"].is_number_integer() || s["repeat_last_n"].get<int>() < 0) {
                return Status::invalid_argument("'repeat_last_n' must be a non-negative integer");
            }
            out_config.sampling.repeat_last_n = s["repeat_last_n"].get<int>();
        }
        if (s.contains("beam_length_penalty")) {
            if (!s["beam_length_penalty"].is_number() || s["beam_length_penalty"].get<float>() <= 0.0f) {
                return Status::invalid_argument("'beam_length_penalty' must be a positive number");
            }
            out_config.sampling.beam_length_penalty = s["beam_length_penalty"].get<float>();
        }
    }

    // Parse limits
    if (j.contains("limits")) {
        const auto &l = j["limits"];
        if (!l.is_object()) return Status::invalid_argument("'limits' must be an object");

        if (l.contains("max_input_bytes")) {
            if (!l["max_input_bytes"].is_number_unsigned()) return Status::invalid_argument("'max_input_bytes' must be unsigned");
            out_config.limits.max_input_bytes = l["max_input_bytes"].get<size_t>();
        }
        if (l.contains("max_multimodal_bytes")) {
            if (!l["max_multimodal_bytes"].is_number_unsigned()) return Status::invalid_argument("'max_multimodal_bytes' must be unsigned");
            out_config.limits.max_multimodal_bytes = l["max_multimodal_bytes"].get<size_t>();
        }
        if (l.contains("max_verbose_bytes")) {
            if (!l["max_verbose_bytes"].is_number_unsigned()) return Status::invalid_argument("'max_verbose_bytes' must be unsigned");
            out_config.limits.max_verbose_bytes = l["max_verbose_bytes"].get<size_t>();
        }
        if (l.contains("max_visual_cache_bytes")) {
            if (!l["max_visual_cache_bytes"].is_number_unsigned()) return Status::invalid_argument("'max_visual_cache_bytes' must be unsigned");
            out_config.limits.max_visual_cache_bytes = l["max_visual_cache_bytes"].get<size_t>();
        }
    }

    // Parse logging
    if (j.contains("logging")) {
        const auto &log_obj = j["logging"];
        if (!log_obj.is_object()) return Status::invalid_argument("'logging' must be an object");
        if (log_obj.contains("level")) {
            if (!log_obj["level"].is_string()) return Status::invalid_argument("'level' must be a string");
            out_config.logging.level = log_obj["level"].get<std::string>();
            static const std::set<std::string> allowed_levels = {"debug", "info", "warn", "error", "off"};
            if (allowed_levels.find(out_config.logging.level) == allowed_levels.end()) {
                return Status::invalid_argument("'level' must be one of: debug, info, warn, error, off");
            }
        }
    }

    return Status::ok();
}

Status load_run_config_file(const std::string &filepath, RunConfig &out_config) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        return Status::io_error("Failed to open run config file: " + filepath);
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    Status st = parse_run_config_json(buffer.str(), out_config);
    if (st && !out_config.vision.mmproj.empty()) {
        auto path = std::filesystem::path(out_config.vision.mmproj);
        if (path.is_relative()) path = std::filesystem::absolute(filepath).parent_path() / path;
        out_config.vision.mmproj = path.lexically_normal().string();
    }
    return st;
}

} // namespace inferlet

#ifndef INFERLET_SUPPORT_JSON_CONFIG_H_INCLUDED
#define INFERLET_SUPPORT_JSON_CONFIG_H_INCLUDED

#include "status.h"
#include <string>
#include <cstdint>
#include <cstddef>

namespace inferlet {

struct RunConfig {
    int schema_version = 1;

    struct ModelSettings {
        int context_size = 4096;
        int n_batch = 512;
        int n_ubatch = 512;
        int threads = 0;
        int threads_batch = 0;
        bool mmap = true;
        bool mlock = false;
        int gpu_layers = 0;
    } model;

    struct VisionSettings {
        std::string mmproj;
    } vision;

    struct SchedulerSettings {
        int worker_count = 0;
        int queue_capacity = 256;
        int stream_queue_chunks = 16;
    } scheduler;

    struct SamplingSettings {
        uint64_t seed = 0;
        int repeat_last_n = 64;
        float beam_length_penalty = 1.0f;
    } sampling;

    struct LimitsSettings {
        size_t max_input_bytes = 67108864; // 64MB
        size_t max_multimodal_bytes = 536870912; // 512MB
        size_t max_verbose_bytes = 536870912; // 512MB
        size_t max_visual_cache_bytes = 1073741824; // 1GB
    } limits;

    struct LoggingSettings {
        std::string level = "info";
    } logging;
};

Status parse_run_config_json(std::string_view json_content, RunConfig &out_config);
Status load_run_config_file(const std::string &filepath, RunConfig &out_config);

} // namespace inferlet

#endif // INFERLET_SUPPORT_JSON_CONFIG_H_INCLUDED

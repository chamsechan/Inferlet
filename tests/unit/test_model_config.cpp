#include "inferlet.h"
#include "core/model_config.h"
#include "core/runtime.h"
#include "support/last_error.h"
#include <cassert>
#include <iostream>
#include <fstream>
#include <unistd.h>

using namespace inferlet;

void test_parameter_defaults() {
    void *param = inferlet_parameter_allocate();
    assert(param != nullptr);

    auto *p = checked_cast<ParameterInternal>(param, ObjectKind::Parameter);
    assert(p != nullptr);
    assert(p->config.valid);
    assert(!p->config.tokenizer_only);
    assert(!p->config.tokenizer_slow);
    assert(p->config.vision_max_patch_cnt == 0);
    assert(p->config.vision_max_grid_size == 0);
    assert(!p->config.vision_transform_skipped);
    assert(p->config.decoder_max_token_count == 512);
    assert(p->config.backend_count == 1);
    assert(!p->config.load_sync);
    assert(p->config.devices.mode == DeviceMode::Auto);

    inferlet_parameter_deallocate(param);
    std::cout << "[PASS] test_parameter_defaults\n";
}

void test_parameter_setters_and_validation() {
    void *param = inferlet_parameter_allocate();
    assert(param != nullptr);
    auto *p = checked_cast<ParameterInternal>(param, ObjectKind::Parameter);

    // Valid setters
    inferlet_parameter_set_tokenizer_only(param, 1);
    assert(p->config.tokenizer_only);
    inferlet_parameter_set_tokenizer_slow(param, 1);
    assert(p->config.tokenizer_slow);
    inferlet_parameter_set_vision_max_patch_cnt(param, 256);
    assert(p->config.vision_max_patch_cnt == 256);
    inferlet_parameter_set_vision_max_grid_size(param, 14);
    assert(p->config.vision_max_grid_size == 14);
    inferlet_parameter_set_vision_trans_skipped(param, 1);
    assert(p->config.vision_transform_skipped);
    inferlet_parameter_set_decoder_max_token_cnt(param, 1024);
    assert(p->config.decoder_max_token_count == 1024);
    inferlet_parameter_set_load_from_file_sync(param, 1);
    assert(p->config.load_sync);
    assert(p->config.valid);

    // Device list NULL auto-generation: count = 4 -> [0, 1, 2, 3]
    inferlet_parameter_set_device_list(param, nullptr, 4);
    assert(p->config.devices.mode == DeviceMode::List);
    assert(p->config.devices.device_list.size() == 4);
    for (int i = 0; i < 4; ++i) {
        assert(p->config.devices.device_list[i] == i);
    }
    assert(p->config.valid);

    // Backend count with device list: 4 devices, backend_cnt = 2 -> 4/2 = 2 (power of 2) -> valid
    inferlet_parameter_set_backend_cnt(param, 2);
    assert(p->config.backend_count == 2);
    assert(p->config.valid);

    // LoRA adapter
    inferlet_parameter_set_low_rank_adapter(param, 1, "test.lora");
    assert(p->config.lora_files.size() == 1);
    assert(p->config.lora_files[1] == "test.lora");
    assert(p->config.valid);

    inferlet_parameter_deallocate(param);
    std::cout << "[PASS] test_parameter_setters_and_validation\n";
}

void test_parameter_invalid_states() {
    // Test invalid tokenizer_only
    {
        void *param = inferlet_parameter_allocate();
        auto *p = checked_cast<ParameterInternal>(param, ObjectKind::Parameter);
        inferlet_parameter_set_tokenizer_only(param, 5);
        assert(!p->config.valid);
        inferlet_parameter_deallocate(param);
    }

    // Test invalid decoder token count
    {
        void *param = inferlet_parameter_allocate();
        auto *p = checked_cast<ParameterInternal>(param, ObjectKind::Parameter);
        inferlet_parameter_set_decoder_max_token_cnt(param, 0);
        assert(!p->config.valid);
        inferlet_parameter_deallocate(param);
    }

    // Test duplicate device IDs in device_list
    {
        void *param = inferlet_parameter_allocate();
        auto *p = checked_cast<ParameterInternal>(param, ObjectKind::Parameter);
        int devs[] = {0, 1, 1};
        inferlet_parameter_set_device_list(param, devs, 3);
        assert(!p->config.valid);
        inferlet_parameter_deallocate(param);
    }

    // Test negative device ID
    {
        void *param = inferlet_parameter_allocate();
        auto *p = checked_cast<ParameterInternal>(param, ObjectKind::Parameter);
        int devs[] = {0, -2};
        inferlet_parameter_set_device_list(param, devs, 2);
        assert(!p->config.valid);
        inferlet_parameter_deallocate(param);
    }

    // Test backend_cnt not dividing device_cnt
    {
        void *param = inferlet_parameter_allocate();
        auto *p = checked_cast<ParameterInternal>(param, ObjectKind::Parameter);
        int devs[] = {0, 1, 2};
        inferlet_parameter_set_device_list(param, devs, 3);
        inferlet_parameter_set_backend_cnt(param, 2); // 3 not divisible by 2
        assert(!p->config.valid);
        inferlet_parameter_deallocate(param);
    }

    // Test backend_cnt dividing device_cnt but quotient not power of 2: e.g. 6 devices, backend_cnt = 2 -> quotient 3
    {
        void *param = inferlet_parameter_allocate();
        auto *p = checked_cast<ParameterInternal>(param, ObjectKind::Parameter);
        int devs[] = {0, 1, 2, 3, 4, 5};
        inferlet_parameter_set_device_list(param, devs, 6);
        inferlet_parameter_set_backend_cnt(param, 2); // 6/2 = 3, not power of 2
        assert(!p->config.valid);
        inferlet_parameter_deallocate(param);
    }

    // Test invalid LoRA ID
    {
        void *param = inferlet_parameter_allocate();
        auto *p = checked_cast<ParameterInternal>(param, ObjectKind::Parameter);
        inferlet_parameter_set_low_rank_adapter(param, 0, "dummy.lora");
        assert(!p->config.valid);
        inferlet_parameter_deallocate(param);
    }

    // Test empty LoRA path
    {
        void *param = inferlet_parameter_allocate();
        auto *p = checked_cast<ParameterInternal>(param, ObjectKind::Parameter);
        inferlet_parameter_set_low_rank_adapter(param, 1, "");
        assert(!p->config.valid);
        inferlet_parameter_deallocate(param);
    }

    std::cout << "[PASS] test_parameter_invalid_states\n";
}

void test_json_run_config() {
    std::string valid_json = R"({
        "schema_version": 1,
        "model": {
            "context_size": 2048,
            "n_batch": 256,
            "n_ubatch": 128,
            "threads": 4,
            "threads_batch": 4,
            "mmap": true,
            "mlock": false,
            "gpu_layers": 0
        },
        "scheduler": {
            "worker_count": 2,
            "queue_capacity": 128,
            "stream_queue_chunks": 8
        },
        "sampling": {
            "seed": 42,
            "repeat_last_n": 32,
            "beam_length_penalty": 1.2
        },
        "limits": {
            "max_input_bytes": 1048576,
            "max_multimodal_bytes": 1048576,
            "max_verbose_bytes": 1048576,
            "max_visual_cache_bytes": 1048576
        },
        "logging": {
            "level": "debug"
        }
    })";

    RunConfig cfg;
    Status st = parse_run_config_json(valid_json, cfg);
    assert(st.is_ok());
    assert(cfg.schema_version == 1);
    assert(cfg.model.context_size == 2048);
    assert(cfg.model.n_batch == 256);
    assert(cfg.scheduler.stream_queue_chunks == 8);
    assert(cfg.sampling.seed == 42);
    assert(cfg.logging.level == "debug");

    // Invalid JSON schema_version
    std::string bad_ver = "{\"schema_version\": 2}";
    assert(!parse_run_config_json(bad_ver, cfg));

    // Unknown top-level key
    std::string unknown_key = "{\"schema_version\": 1, \"foo\": 123}";
    assert(!parse_run_config_json(unknown_key, cfg));

    // Broken JSON syntax
    std::string broken = "{\"schema_version\": ";
    assert(!parse_run_config_json(broken, cfg));

    std::string invalid_log_level = R"({"schema_version": 1, "logging": {"level": "trace"}})";
    assert(!parse_run_config_json(invalid_log_level, cfg));

    std::cout << "[PASS] test_json_run_config\n";
}

void test_supported_config_and_fingerprint() {
    ModelConfig baseline;
    assert(baseline.validate_supported().is_ok());

    ModelConfig slow_tokenizer = baseline;
    slow_tokenizer.tokenizer_slow = true;
    assert(slow_tokenizer.validate_supported().code == INFERLET_E_UNSUPPORTED);

    ModelConfig vision = baseline;
    vision.vision_max_patch_cnt = 128;
    assert(vision.validate_supported().code == INFERLET_E_UNSUPPORTED);

    ModelConfig workers = baseline;
    workers.parsed_run_config.scheduler.worker_count = 2;
    assert(workers.validate_supported().code == INFERLET_E_UNSUPPORTED);

    ModelConfig different_ubatch = baseline;
    different_ubatch.parsed_run_config.model.n_ubatch = 64;
    assert(different_ubatch.compute_fingerprint() != baseline.compute_fingerprint());

    ModelConfig different_backend_count = baseline;
    different_backend_count.backend_count = 2;
    assert(different_backend_count.compute_fingerprint() != baseline.compute_fingerprint());

    std::cout << "[PASS] test_supported_config_and_fingerprint\n";
}

void test_config_priority_and_merging() {
    // Write a temporary valid run_config.json
    const std::string tmp_file = "/tmp/test_inferlet_run_config.json";
    {
        std::ofstream ofs(tmp_file);
        ofs << R"({
            "schema_version": 1,
            "model": {
                "n_batch": 1024
            }
        })";
    }

    // Case 1: Setter overrides JSON
    {
        void *param = inferlet_parameter_allocate();
        inferlet_parameter_set_run_config_file(param, tmp_file.c_str());
        inferlet_parameter_set_decoder_max_token_cnt(param, 2048);

        auto *p = checked_cast<ParameterInternal>(param, ObjectKind::Parameter);
        Status st = p->config.load_and_merge_run_config();
        assert(st.is_ok());
        // Setter had 2048, so it stays 2048
        assert(p->config.decoder_max_token_count == 2048);
        inferlet_parameter_deallocate(param);
    }

    // Case 2: JSON defaults apply if setter was not called
    {
        void *param = inferlet_parameter_allocate();
        inferlet_parameter_set_run_config_file(param, tmp_file.c_str());

        auto *p = checked_cast<ParameterInternal>(param, ObjectKind::Parameter);
        Status st = p->config.load_and_merge_run_config();
        assert(st.is_ok());
        // JSON had 1024, so it overrides library default 512
        assert(p->config.decoder_max_token_count == 1024);
        inferlet_parameter_deallocate(param);
    }

    unlink(tmp_file.c_str());
    std::cout << "[PASS] test_config_priority_and_merging\n";
}

void test_load_from_file_with_invalid_param() {
    inferlet_init();

    void *param = inferlet_parameter_allocate();
    // Invalidate parameter
    inferlet_parameter_set_decoder_max_token_cnt(param, -10);

    void *handle = inferlet_load_from_file("dummy.gguf", param);
    assert(handle == nullptr);
    assert(get_last_error_code() == INFERLET_E_INVALID_ARGUMENT);

    inferlet_parameter_deallocate(param);
    inferlet_deinit();
    std::cout << "[PASS] test_load_from_file_with_invalid_param\n";
}

void test_vision_config() {
    RunConfig cfg;
    assert(parse_run_config_json(R"({"schema_version":1,"vision":{"mmproj":"model.gguf"}})", cfg));
    assert(cfg.vision.mmproj == "model.gguf");
    assert(!parse_run_config_json(R"({"schema_version":1,"vision":[]})", cfg));
    assert(!parse_run_config_json(R"({"schema_version":1,"vision":{"mmproj":""}})", cfg));
    assert(!parse_run_config_json(R"({"schema_version":1,"vision":{"mmproj":42}})", cfg));
    assert(!parse_run_config_json(R"({"schema_version":1,"vision":{"mmproj":"a","unknown":true}})", cfg));
    ModelConfig baseline, vision;
    vision.parsed_run_config.vision.mmproj = "model.gguf";
    assert(vision.validate_supported());
    assert(baseline.compute_fingerprint() != vision.compute_fingerprint());
    vision.tokenizer_only = true;
    assert(vision.validate_supported().code == INFERLET_E_UNSUPPORTED);
    vision.tokenizer_only = false;
    vision.vision_max_patch_cnt = 64;
    assert(vision.validate_supported().code == INFERLET_E_UNSUPPORTED);
    const std::string path = "/tmp/test_inferlet_vision_config.json";
    { std::ofstream file(path); file << R"({"schema_version":1,"vision":{"mmproj":"models/vision.gguf"}})"; }
    RunConfig relative;
    assert(load_run_config_file(path, relative));
    assert(relative.vision.mmproj == "/tmp/models/vision.gguf");
    unlink(path.c_str());
}

int main() {
    test_parameter_defaults();
    test_parameter_setters_and_validation();
    test_parameter_invalid_states();
    test_json_run_config();
    test_vision_config();
    test_supported_config_and_fingerprint();
    test_config_priority_and_merging();
    test_load_from_file_with_invalid_param();

    std::cout << "All ModelConfig and Parameter tests passed successfully!\n";
    return 0;
}

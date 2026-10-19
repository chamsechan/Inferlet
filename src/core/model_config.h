#ifndef INFERLET_CORE_MODEL_CONFIG_H_INCLUDED
#define INFERLET_CORE_MODEL_CONFIG_H_INCLUDED

#include "support/object_registry.h"
#include "support/json_config.h"
#include "support/status.h"
#include <string>
#include <vector>
#include <map>
#include <set>

namespace inferlet {

enum class DeviceMode {
    Auto,
    Single,
    List
};

struct DeviceConfig {
    DeviceMode mode = DeviceMode::Auto;
    int single_device_id = -1;
    std::vector<int> device_list;
};

enum class ConfigField {
    TokenizerOnly,
    TokenizerSlow,
    VisionMaxPatchCnt,
    VisionMaxGridSize,
    VisionTransformSkipped,
    DecoderMaxTokenCount,
    DeviceId,
    DeviceList,
    BackendCount,
    LowRankAdapter,
    LoadFromFileSync,
    RunConfigFile
};

struct ModelConfig {
    bool tokenizer_only = false;
    bool tokenizer_slow = false;
    int vision_max_patch_cnt = 0;
    int vision_max_grid_size = 0;
    bool vision_transform_skipped = false;
    int decoder_max_token_count = 512;

    DeviceConfig devices;
    int backend_count = 1;
    std::map<int, std::string> lora_files;
    bool load_sync = false;
    std::string run_config_file;

    RunConfig parsed_run_config;
    std::set<ConfigField> explicitly_set;
    bool valid = true;
    std::string invalid_reason;

    void mark_invalid(std::string reason);
    std::string compute_fingerprint() const;
    Status load_and_merge_run_config();
    Status validate_supported() const;
};

struct ParameterInternal {
    ObjectHeader header{INFERLET_OBJECT_MAGIC, ObjectKind::Parameter, INFERLET_ABI_VERSION};
    ModelConfig config;

    void set_tokenizer_only(int val);
    void set_tokenizer_slow(int val);
    void set_vision_max_patch_cnt(int val);
    void set_vision_max_grid_size(int val);
    void set_vision_trans_skipped(int val);
    void set_decoder_max_token_cnt(int val);
    void set_device_id(int val);
    void set_device_list(const int *device_list, int device_cnt);
    void set_backend_cnt(int val);
    void set_low_rank_adapter(int id, const char *file);
    void set_load_from_file_sync(int val);
    void set_run_config_file(const char *file);
};

} // namespace inferlet

#endif // INFERLET_CORE_MODEL_CONFIG_H_INCLUDED

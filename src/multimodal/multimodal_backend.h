#ifndef INFERLET_MULTIMODAL_MULTIMODAL_BACKEND_H_INCLUDED
#define INFERLET_MULTIMODAL_MULTIMODAL_BACKEND_H_INCLUDED

#include "support/status.h"
#include "core/task_input.h"
#include <string>
#include <vector>
#include <memory>
#include <variant>
#include <cstdint>

namespace inferlet {

enum class MediaType {
    Text,
    Image,
    Video,
    Audio
};

enum class DataType {
    Undef = 0,
    Float = 1,
    Half = 2,
    BF16 = 3,
    UInt8 = 4
};

struct FrameInfo {
    int frame_num = 0;
    int packing_num = 1;
    int slice_num = 0;
    std::vector<int> frame_index;
    std::string language;
};

struct CachePolicy {
    std::string key;
    int cache_times = 0;
};

struct MediaTensor {
    MediaType type = MediaType::Image;
    DataType dtype = DataType::Float;
    std::vector<uint8_t> owned_data;
    int grid_t = 1;
    int grid_h = 1;
    int grid_w = 1;
    float seconds_per_grid_t = 0.0f;
    float fps = 24.0f;
    FrameInfo frames;
    CachePolicy cache;
};

struct MessagePart {
    std::variant<std::string, MediaTensor> value;
};

struct VisualEncoderMetadata {
    std::string id;
    int channels = 3;
    int patch_size = 14;
    int temporal_patch_size = 2;
    int n_embd = 896;
};

struct EncodedVisualFeature {
    std::shared_ptr<const std::vector<float>> embeddings;
    int n_tokens = 0;
    int n_embd = 896;
};

class MultimodalBackend {
public:
    virtual ~MultimodalBackend() = default;

    virtual Status get_encoder_metadata(const std::string &encoder_id, VisualEncoderMetadata &out_meta) const = 0;

    virtual Status encode(const std::string &encoder_id,
                          const std::string &model_fingerprint,
                          const MediaTensor &tensor,
                          EncodedVisualFeature &out_feature) = 0;

    static bool checked_mul(size_t a, size_t b, size_t &out_result);

    static Status validate_tensor(const MediaTensor &tensor, const VisualEncoderMetadata &meta);

    static Status normalize_task_input(const TaskRequest &request,
                                       const VisualEncoderMetadata &meta,
                                       std::vector<MessagePart> &out_parts);
};

} // namespace inferlet

#endif // INFERLET_MULTIMODAL_MULTIMODAL_BACKEND_H_INCLUDED

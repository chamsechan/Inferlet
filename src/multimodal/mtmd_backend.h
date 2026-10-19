#ifndef INFERLET_MULTIMODAL_MTMD_BACKEND_H_INCLUDED
#define INFERLET_MULTIMODAL_MTMD_BACKEND_H_INCLUDED

#include "core/model_config.h"
#include "visual_cache.h"
#include "mtmd.h"
#include <atomic>
#include <mutex>

namespace inferlet {
struct TaskRequest;
struct SharedModel;
struct PreparedPrompt;

struct MtmdPrompt {
    struct Image {
        std::string cache_key;
        uint64_t hash = 0;
        int grid_h = 0, grid_w = 0, cache_times = 0;
    };
    std::vector<Image> images;
    std::vector<mtmd::bitmap_ptr> bitmaps;
    mtmd::input_chunks_ptr chunks{mtmd_input_chunks_init()};
};

class MtmdBackend {
public:
    static Status load(const ModelConfig &config, const llama_model *model,
                       std::shared_ptr<MtmdBackend> &out);
    Status prepare(const TaskRequest &request, const SharedModel &model, PreparedPrompt &out);
    Status prefill(const MtmdPrompt &prompt, llama_context *ctx,
                   const std::atomic<bool> *cancel, int &next_position);
private:
    explicit MtmdBackend(size_t cache_bytes) : cache_(cache_bytes) {}
    mtmd::context_ptr ctx_;
    int patch_size_ = 0;
    int n_embd_ = 0;
    VisualCache cache_;
    // libmtmd's encoder output buffer is shared and must be serialized.
    std::mutex mutex_;
};
} // namespace inferlet
#endif

#include "multimodal_backend.h"
#include <limits>
#include <cstring>

namespace inferlet {

bool MultimodalBackend::checked_mul(size_t a, size_t b, size_t &out_result) {
    if (a == 0 || b == 0) {
        out_result = 0;
        return true;
    }
    if (a > std::numeric_limits<size_t>::max() / b) {
        return false;
    }
    out_result = a * b;
    return true;
}

Status MultimodalBackend::validate_tensor(const MediaTensor &tensor, const VisualEncoderMetadata &meta) {
    if (tensor.type == MediaType::Image || tensor.type == MediaType::Video) {
        if (tensor.owned_data.empty()) {
            return Status::invalid_argument("Visual media tensor data cannot be empty");
        }
        if (tensor.grid_t <= 0 || tensor.grid_h <= 0 || tensor.grid_w <= 0) {
            return Status::invalid_argument("grid_t, grid_h, grid_w must all be > 0");
        }

        int dt = static_cast<int>(tensor.dtype);
        if (dt < 1 || dt > 4) {
            return Status::invalid_argument("Media tensor data_type must be in 1..4 (1=float, 2=half, 3=bf16, 4=uint8)");
        }

        if (!tensor.cache.key.empty() && tensor.cache.cache_times < 2) {
            return Status::invalid_argument("cache_times must be >= 2 when cache_key is non-empty");
        }

        if (tensor.frames.frame_num < 0) {
            return Status::invalid_argument("frame_num cannot be negative");
        }

        if (tensor.frames.packing_num < 1 || tensor.frames.packing_num > 6) {
            return Status::invalid_argument("packing_num must be between 1 and 6");
        }

        if (tensor.fps < 0.0f) {
            return Status::invalid_argument("fps cannot be negative");
        }

        if (tensor.dtype == DataType::Float || tensor.dtype == DataType::Half || tensor.dtype == DataType::BF16) {
            size_t elem_size = (tensor.dtype == DataType::Float) ? 4 : 2;
            size_t n_patches = 0;
            if (!checked_mul(tensor.grid_t, tensor.grid_h, n_patches)) {
                return Status::invalid_argument("Integer overflow computing grid patches");
            }
            if (!checked_mul(n_patches, tensor.grid_w, n_patches)) {
                return Status::invalid_argument("Integer overflow computing grid patches");
            }

            size_t total_elements = 0;
            if (!checked_mul(n_patches, meta.n_embd, total_elements)) {
                return Status::invalid_argument("Integer overflow computing total embedding elements");
            }

            size_t expected_size = 0;
            if (!checked_mul(total_elements, elem_size, expected_size)) {
                return Status::invalid_argument("Integer overflow computing expected feature byte size");
            }

            if (tensor.owned_data.size() != expected_size) {
                return Status::invalid_argument("Feature byte size (" + std::to_string(tensor.owned_data.size()) +
                                               ") does not match expected size (" + std::to_string(expected_size) +
                                               " = grid_t*grid_h*grid_w*n_embd*element_size)");
            }
        } else if (tensor.dtype == DataType::UInt8) {
            size_t t = (tensor.type == MediaType::Video) ?
                       static_cast<size_t>(tensor.grid_t) * meta.temporal_patch_size : 1;
            size_t h = static_cast<size_t>(tensor.grid_h) * meta.patch_size;
            size_t w = static_cast<size_t>(tensor.grid_w) * meta.patch_size;
            size_t c = meta.channels;

            size_t expected_size = 0;
            if (!checked_mul(t, c, expected_size)) return Status::invalid_argument("Integer overflow in raw image shape");
            if (!checked_mul(expected_size, h, expected_size)) return Status::invalid_argument("Integer overflow in raw image shape");
            if (!checked_mul(expected_size, w, expected_size)) return Status::invalid_argument("Integer overflow in raw image shape");

            if (tensor.owned_data.size() != expected_size) {
                return Status::invalid_argument("Raw uint8 image size (" + std::to_string(tensor.owned_data.size()) +
                                               ") does not match expected shape size (" + std::to_string(expected_size) +
                                               " = t*c*h*w)");
            }
        }
    }

    return Status::ok();
}

Status MultimodalBackend::normalize_task_input(const TaskRequest &request,
                                               const VisualEncoderMetadata &meta,
                                               std::vector<MessagePart> &out_parts) {
    out_parts.clear();

    if (std::holds_alternative<LegacyMultiModalInput>(request.input)) {
        const auto &lmi = std::get<LegacyMultiModalInput>(request.input);
        for (const auto &item : lmi.items) {
            if (!item.prompt_text.empty()) {
                MessagePart p;
                p.value = item.prompt_text;
                out_parts.push_back(std::move(p));
            }
            if (!item.feature.empty()) {
                size_t feature_bytes = item.feature.size();
                size_t token_bytes = static_cast<size_t>(meta.n_embd) * sizeof(float);
                if (feature_bytes % token_bytes != 0) {
                    return Status::invalid_argument("Legacy multimodal feature size (" + std::to_string(feature_bytes) +
                                                   ") must be a multiple of n_embd * sizeof(float) (" +
                                                   std::to_string(token_bytes) + ")");
                }
                size_t n_tokens = feature_bytes / token_bytes;

                MediaTensor tensor;
                tensor.type = MediaType::Image;
                tensor.dtype = DataType::Float;
                tensor.owned_data = item.feature;
                tensor.grid_t = 1;
                tensor.grid_h = 1;
                tensor.grid_w = static_cast<int>(n_tokens);

                Status st = validate_tensor(tensor, meta);
                if (!st) return st;

                MessagePart p;
                p.value = std::move(tensor);
                out_parts.push_back(std::move(p));
            }
        }
        return Status::ok();
    }

    if (std::holds_alternative<MultiModalChatInput>(request.input)) {
        const auto &mmci = std::get<MultiModalChatInput>(request.input);
        int common_dtype = -1;

        for (const auto &msg : mmci.history) {
            for (const auto &content : msg.contents) {
                if (content.type == "text") {
                    MessagePart p;
                    p.value = content.text_data;
                    out_parts.push_back(std::move(p));
                } else if (content.type == "image" || content.type == "video") {
                    if (common_dtype == -1) {
                        common_dtype = content.data_type;
                    } else if (common_dtype != content.data_type) {
                        return Status::invalid_argument("MultiModal data_type must be consistent across entire task");
                    }

                    MediaTensor tensor;
                    tensor.type = (content.type == "video") ? MediaType::Video : MediaType::Image;
                    tensor.dtype = static_cast<DataType>(content.data_type);
                    tensor.owned_data = content.binary_data;
                    tensor.grid_t = content.grid_t;
                    tensor.grid_h = content.grid_h;
                    tensor.grid_w = content.grid_w;
                    tensor.seconds_per_grid_t = content.second_per_grid_t;
                    tensor.fps = content.fps;

                    tensor.frames.frame_num = content.ext.frame_num;
                    tensor.frames.packing_num = content.ext.packing_num == 0 ? 1 : content.ext.packing_num;
                    tensor.frames.slice_num = content.ext.slice_num;
                    tensor.frames.frame_index = content.ext.frame_index;
                    tensor.frames.language = content.ext.language;

                    tensor.cache.key = content.cache_key;
                    tensor.cache.cache_times = content.cache_times;

                    Status st = validate_tensor(tensor, meta);
                    if (!st) return st;

                    MessagePart p;
                    p.value = std::move(tensor);
                    out_parts.push_back(std::move(p));
                } else {
                    return Status::invalid_argument("Unsupported multimodal chat content type: " + content.type);
                }
            }
        }
        return Status::ok();
    }

    return Status::ok();
}

} // namespace inferlet

#include "mtmd_backend.h"
#include "engine/prompt_builder.h"
#include "engine/model_prompt_adapter.h"
#include "mtmd-helper.h"
#include "gguf.h"
#include <limits>
#include <unordered_map>

namespace inferlet {
Status MtmdBackend::load(const ModelConfig &config, const llama_model *model,
                         std::shared_ptr<MtmdBackend> &out) {
    try {
        const auto &path = config.parsed_run_config.vision.mmproj;
        std::unique_ptr<gguf_context, decltype(&gguf_free)> meta(
            gguf_init_from_file(path.c_str(), {true, nullptr}), gguf_free);
        if (!meta) return Status::model_load("Cannot read visual projector: " + path);
        const auto key = gguf_find_key(meta.get(), "clip.vision.patch_size");
        if (key < 0 || gguf_get_kv_type(meta.get(), key) != GGUF_TYPE_UINT32) {
            return Status::model_load("Visual projector requires uint32 clip.vision.patch_size metadata");
        }
        const uint32_t patch = gguf_get_val_u32(meta.get(), key);
        if (patch == 0 || patch > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
            return Status::model_load("Invalid visual projector patch size");
        }
        auto backend = std::shared_ptr<MtmdBackend>(new MtmdBackend(config.parsed_run_config.limits.max_visual_cache_bytes));
        backend->patch_size_ = static_cast<int>(patch);
        backend->n_embd_ = llama_model_n_embd_inp(model);
        auto params = mtmd_context_params_default();
        params.use_gpu = false; // CPU baseline; device-specific vision execution is not exposed yet.
        params.n_threads = config.parsed_run_config.model.threads > 0 ? config.parsed_run_config.model.threads : 4;
        params.warmup = false;
        backend->ctx_.reset(mtmd_init_from_file(path.c_str(), model, params));
        if (!backend->ctx_ || !mtmd_support_vision(backend->ctx_.get())) {
            return Status::model_load("Cannot load a compatible vision projector: " + path);
        }
        out = std::move(backend);
        return Status::ok();
    } catch (const std::bad_alloc &) {
        return Status::out_of_memory("Allocating visual projector failed");
    } catch (const std::exception &e) {
        return Status::model_load(std::string("Loading visual projector failed: ") + e.what());
    }
}

Status MtmdBackend::prepare(const TaskRequest &request, const SharedModel &model, PreparedPrompt &out) try {
    const auto *chat = std::get_if<MultiModalChatInput>(&request.input);
    if (!chat) return Status::unsupported("Use MultiModal_ChatHistory with raw RGB images; legacy features are unsupported");
    if (!request.visual_encoder_id.empty() && request.visual_encoder_id != "default") {
        return Status::unsupported("Only the default visual encoder is available");
    }
    if (chat->history.empty() || chat->history.back().role != "user") {
        return Status::invalid_argument("Multimodal chat must end with a user message");
    }
    auto prep = std::make_shared<MtmdPrompt>();
    std::vector<std::pair<std::string, std::string>> messages;
    std::vector<const mtmd_bitmap *> bitmaps;
    const std::string marker = mtmd_default_marker();
    for (const auto &message : chat->history) {
        std::string text;
        for (const auto &content : message.contents) {
            if (content.type == "text") {
                if (content.text_data.find(marker) != std::string::npos) {
                    return Status::invalid_argument("Text contains the reserved image marker");
                }
                text += content.text_data;
                continue;
            }
            if (content.type != "image" || content.data_type != 4) {
                return Status::unsupported("Vision input currently supports UINT8 RGB images only");
            }
            if (content.grid_t != 1 || content.ext.frame_num != 0 || !content.ext.frame_index.empty() ||
                content.ext.packing_num != 1 || content.ext.slice_num != 0 || !content.ext.language.empty()) {
                return Status::unsupported("Image frame, packing and slice extensions are unsupported");
            }
            const uint64_t h = static_cast<uint64_t>(content.grid_h) * patch_size_;
            const uint64_t w = static_cast<uint64_t>(content.grid_w) * patch_size_;
            if (content.grid_h <= 0 || content.grid_w <= 0 ||
                h > std::numeric_limits<int>::max() || w > std::numeric_limits<int>::max() ||
                h * w > content.binary_data.size() / 3 || h * w * 3 != content.binary_data.size()) {
                return Status::invalid_argument("Image size must equal 3 * grid_h * grid_w * patch_size^2 bytes (planar RGB)");
            }
            const size_t pixels = static_cast<size_t>(h * w);
            std::vector<unsigned char> rgb(pixels * 3);
            for (size_t i = 0; i < pixels; ++i) {
                for (size_t c = 0; c < 3; ++c) rgb[3 * i + c] = content.binary_data[c * pixels + i];
            }
            mtmd::bitmap_ptr bitmap(mtmd_bitmap_init(static_cast<uint32_t>(w), static_cast<uint32_t>(h), rgb.data()));
            if (!bitmap) return Status::out_of_memory("Creating image bitmap failed");
            const std::string id = std::to_string(prep->images.size());
            mtmd_bitmap_set_id(bitmap.get(), id.c_str());
            prep->images.push_back({content.cache_key,
                VisualCache::compute_hash(content.binary_data.data(), content.binary_data.size()),
                content.grid_h, content.grid_w, content.cache_times});
            bitmaps.push_back(bitmap.get());
            prep->bitmaps.push_back(std::move(bitmap));
            text += marker;
        }
        messages.emplace_back(message.role, std::move(text));
    }
    if (!request.tools.empty()) {
        auto tools = ModelPromptAdapter::format_tools_description(request.tools);
        if (tools.find(marker) != std::string::npos) return Status::invalid_argument("Tools contain the reserved image marker");
        if (messages.front().first == "system") messages.front().second += "\n\n" + tools;
        else messages.insert(messages.begin(), {"system", std::move(tools)});
    }
    const char *tmpl = LlamaEngine::get_model_chat_template(model.model.get());
    if (!tmpl || LlamaEngine::apply_chat_template(tmpl, messages, true, out.formatted_text) < 0) {
        return Status::unsupported("Cannot apply this model's multimodal chat template");
    }
    mtmd_input_text input{out.formatted_text.c_str(), out.formatted_text.size(), false, true};
    std::lock_guard<std::mutex> lock(mutex_);
    if (mtmd_tokenize(ctx_.get(), prep->chunks.get(), &input, bitmaps.data(), bitmaps.size()) != 0) {
        return Status::invalid_argument("Image preprocessing or image marker alignment failed");
    }
    const size_t count = mtmd_helper_get_n_tokens(prep->chunks.get());
    if (count == 0 || count >= static_cast<size_t>(model.config.parsed_run_config.model.context_size)) {
        return Status::invalid_argument("Image and text tokens exceed the configured context size");
    }
    const size_t chunks = mtmd_input_chunks_size(prep->chunks.get());
    if (chunks == 0 || mtmd_input_chunk_get_type(mtmd_input_chunks_get(prep->chunks.get(), chunks - 1)) != MTMD_INPUT_CHUNK_TYPE_TEXT) {
        return Status::unsupported("Multimodal template must include assistant text after the last image");
    }
    out.tokens.reserve(count);
    for (size_t i = 0; i < chunks; ++i) {
        const auto *chunk = mtmd_input_chunks_get(prep->chunks.get(), i);
        if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t n = 0;
            const auto *tokens = mtmd_input_chunk_get_tokens_text(chunk, &n);
            if (n) out.tokens.insert(out.tokens.end(), tokens, tokens + n);
        } else {
            // Image slots are not vocabulary tokens and must never enter the text sampler.
            out.tokens.insert(out.tokens.end(), mtmd_input_chunk_get_n_tokens(chunk), -1);
        }
    }
    out.multimodal = std::move(prep);
    return Status::ok();
} catch (const std::bad_alloc &) {
    return Status::out_of_memory("Image preprocessing allocation failed");
} catch (const std::exception &e) {
    return Status::inference_error(std::string("Image preprocessing failed: ") + e.what());
}

Status MtmdBackend::prefill(const MtmdPrompt &prompt, llama_context *lctx,
                           const std::atomic<bool> *cancel, int &next_position) try {
    std::lock_guard<std::mutex> lock(mutex_);
    llama_pos position = 0;
    const int batch = static_cast<int>(llama_n_batch(lctx));
    const size_t count = mtmd_input_chunks_size(prompt.chunks.get());
    std::unordered_map<std::string, size_t> slices;
    for (size_t i = 0; i < count; ++i) {
        if (cancel && cancel->load()) return Status::cancelled();
        const auto *chunk = mtmd_input_chunks_get(prompt.chunks.get(), i);
        int rc = 0;
        if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            rc = mtmd_helper_eval_chunk_single(ctx_.get(), lctx, chunk, position, 0, batch, i + 1 == count, &position);
        } else if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            const size_t tokens = mtmd_input_chunk_get_n_tokens(chunk);
            if (mtmd_decode_use_non_causal(ctx_.get(), chunk) && tokens > llama_n_ubatch(lctx)) {
                return Status::invalid_argument("This image requires model.n_batch and model.n_ubatch >= its visual token count");
            }
            const char *id = mtmd_input_chunk_get_id(chunk);
            if (!id) return Status::internal("Image chunk has no source ID");
            const auto index = static_cast<size_t>(std::stoull(id));
            if (index >= prompt.images.size()) return Status::internal("Invalid image chunk source ID");
            const auto &image = prompt.images[index];
            const auto slice = slices[id]++;
            const auto key = std::to_string(image.cache_key.size()) + ":" + image.cache_key + ":" + std::to_string(slice);
            std::shared_ptr<const std::vector<float>> embeddings;
            if (!image.cache_key.empty()) {
                Status st = cache_.lookup(key, image.hash, 4, 1, image.grid_h, image.grid_w, embeddings);
                if (!st && st.code != INFERLET_E_IO) return st;
            }
            if (!embeddings) {
                if (mtmd_encode_chunk(ctx_.get(), chunk) != 0) return Status::inference_error("Visual encoder failed");
                const float *data = mtmd_get_output_embd(ctx_.get());
                if (!data) return Status::inference_error("Visual encoder produced no embeddings");
                embeddings = std::make_shared<std::vector<float>>(data, data + tokens * n_embd_);
                if (!image.cache_key.empty()) {
                    Status st = cache_.insert(key, image.hash, 4, 1, image.grid_h, image.grid_w, embeddings, image.cache_times);
                    if (!st) return st;
                }
            }
            if (embeddings->size() != tokens * n_embd_) return Status::internal("Cached image embedding size mismatch");
            if (cancel && cancel->load()) return Status::cancelled();
            // Upstream accepts float* but only reads it while constructing llama batches.
            rc = mtmd_helper_decode_image_chunk(ctx_.get(), lctx, chunk, const_cast<float *>(embeddings->data()),
                                               position, 0, batch, &position, nullptr, nullptr);
        } else {
            return Status::unsupported("Audio chunks are unsupported");
        }
        if (rc != 0) return cancel && cancel->load() ? Status::cancelled() : Status::inference_error("Multimodal prompt decoding failed");
    }
    next_position = position; // M-RoPE positions can differ from the number of image tokens.
    return Status::ok();
} catch (const std::bad_alloc &) {
    return Status::out_of_memory("Image encoding allocation failed");
} catch (const std::exception &e) {
    return Status::inference_error(std::string("Image encoding failed: ") + e.what());
}
} // namespace inferlet

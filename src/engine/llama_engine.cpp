#include "llama_engine.h"
#include "core/model_registry.h"
#include "core/task_input.h"
#include "engine/lora_registry.h"
#include "engine/model_prompt_adapter.h"
#include "support/logging.h"
#include <llama.h>
#include <filesystem>
#include <thread>
#include <algorithm>
#include <cctype>

namespace inferlet {

[[maybe_unused]] constexpr const char *INFERLET_LLAMA_COMMIT = "8b4b3558f1459c13e4aa38d5c94d306a00dc6acd";
static_assert(sizeof(llama_token) == sizeof(int), "llama_token size must match int size");

void LlamaModelDeleter::operator()(llama_model *m) const {
    if (m) {
        log_debug("Freeing llama_model");
        llama_model_free(m);
    }
}

void LlamaContextDeleter::operator()(llama_context *c) const {
    if (c) {
        log_debug("Freeing llama_context");
        llama_free(c);
    }
}

void LlamaLoraDeleter::operator()(llama_adapter_lora *l) const {
    if (l) {
        log_debug("Freeing llama_adapter_lora");
        llama_adapter_lora_free(l);
    }
}

void LlamaSamplerDeleter::operator()(llama_sampler *s) const {
    if (s) {
        log_debug("Freeing llama_sampler");
        llama_sampler_free(s);
    }
}

void global_llama_backend_init() {
    llama_backend_init();
}

void global_llama_backend_free() {
    llama_backend_free();
}

Status LlamaEngine::load_model(const ModelConfig &config,
                               const std::string &model_path,
                               LlamaModelPtr &out_model,
                               const llama_vocab *&out_vocab) {
    std::error_code ec;
    if (!std::filesystem::exists(model_path, ec)) {
        return Status::model_load("Model file does not exist: " + model_path);
    }

    llama_model_params mparams = llama_model_default_params();
    mparams.vocab_only = config.tokenizer_only;

    if (config.parsed_run_config.model.mmap && config.parsed_run_config.model.mlock) {
        mparams.load_mode = LLAMA_LOAD_MODE_MMAP_MLOCK;
    } else if (config.parsed_run_config.model.mmap) {
        mparams.load_mode = LLAMA_LOAD_MODE_MMAP;
    } else if (config.parsed_run_config.model.mlock) {
        mparams.load_mode = LLAMA_LOAD_MODE_MLOCK;
    } else {
        mparams.load_mode = LLAMA_LOAD_MODE_NONE;
    }

    std::vector<ggml_backend_dev_t> selected_devices;
    auto append_device = [&](int device_id) -> Status {
        const size_t device_count = ggml_backend_dev_count();
        if (device_id < 0 || static_cast<size_t>(device_id) >= device_count) {
            return Status::invalid_argument("Device ID " + std::to_string(device_id) +
                                            " is outside available range [0, " +
                                            std::to_string(device_count) + ")");
        }
        selected_devices.push_back(ggml_backend_dev_get(static_cast<size_t>(device_id)));
        return Status::ok();
    };
    if (config.devices.mode == DeviceMode::Single) {
        Status device_st = append_device(config.devices.single_device_id);
        if (!device_st) return device_st;
    } else if (config.devices.mode == DeviceMode::List) {
        for (int device_id : config.devices.device_list) {
            Status device_st = append_device(device_id);
            if (!device_st) return device_st;
        }
    }
    if (!selected_devices.empty()) {
        selected_devices.push_back(nullptr);
        mparams.devices = selected_devices.data();
        mparams.main_gpu = 0;
    }

    // Apply run config if loaded
    if (config.parsed_run_config.model.gpu_layers > 0) {
        mparams.n_gpu_layers = config.parsed_run_config.model.gpu_layers;
    }

    log_info("Loading model from: " + model_path + " (vocab_only=" + std::to_string(mparams.vocab_only) + ")");

    llama_model *model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        return Status::model_load("llama_model_load_from_file failed for: " + model_path);
    }

    out_model.reset(model);
    out_vocab = llama_model_get_vocab(model);
    if (!out_vocab) {
        out_model.reset();
        return Status::model_load("Failed to extract vocab from model: " + model_path);
    }

    return Status::ok();
}

Status LlamaEngine::load_lora(llama_model *model,
                              const std::string &lora_path,
                              LlamaLoraPtr &out_lora) {
    if (!model) {
        return Status::invalid_handle("Base model is NULL for LoRA adapter");
    }
    std::error_code ec;
    if (!std::filesystem::exists(lora_path, ec)) {
        return Status::io_error("LoRA file does not exist: " + lora_path);
    }
    llama_adapter_lora *lora = llama_adapter_lora_init(model, lora_path.c_str());
    if (!lora) {
        return Status::model_load("Failed to load LoRA adapter from: " + lora_path);
    }
    out_lora.reset(lora);
    return Status::ok();
}

const char *LlamaEngine::get_model_chat_template(const llama_model *model, const char *name) {
    if (!model) return nullptr;
    return llama_model_chat_template(model, name);
}

int32_t LlamaEngine::apply_chat_template(const char *tmpl,
                                         const std::vector<std::pair<std::string, std::string>> &chat,
                                         bool add_ass,
                                         std::string &out_formatted) {
    std::vector<llama_chat_message> msgs;
    msgs.reserve(chat.size());
    for (const auto &item : chat) {
        msgs.push_back({item.first.c_str(), item.second.c_str()});
    }

    int32_t needed = llama_chat_apply_template(tmpl, msgs.data(), msgs.size(), add_ass, nullptr, 0);
    if (needed < 0) {
        return needed;
    }

    std::vector<char> buf(needed + 1, '\0');
    int32_t written = llama_chat_apply_template(tmpl, msgs.data(), msgs.size(), add_ass, buf.data(), buf.size());
    if (written >= 0) {
        out_formatted.assign(buf.data(), written);
    }
    return written;
}

int LlamaEngine::get_vocab_bos(const llama_vocab *vocab) {
    if (!vocab) return -1;
    return llama_vocab_bos(vocab);
}

int LlamaEngine::get_vocab_eos(const llama_vocab *vocab) {
    if (!vocab) return -1;
    return llama_vocab_eos(vocab);
}

int32_t LlamaEngine::get_vocab_n_tokens(const llama_vocab *vocab) {
    if (!vocab) return 0;
    return llama_vocab_n_tokens(vocab);
}

bool LlamaEngine::is_vocab_eog(const llama_vocab *vocab, int token) {
    if (!vocab) return false;
    return llama_vocab_is_eog(vocab, token);
}

Status LlamaEngine::create_context(const SharedModel &model,
                                   const TaskRequest &request,
                                   uint32_t prompt_tokens_count,
                                   LlamaContextPtr &out_ctx) {
    if (!model.model) {
        return Status::invalid_handle("Model is NULL in SharedModel");
    }

    llama_context_params cparams = llama_context_default_params();
    uint32_t max_allowed_ctx = 0;
    if (model.config.parsed_run_config.model.context_size > 0) {
        max_allowed_ctx = static_cast<uint32_t>(model.config.parsed_run_config.model.context_size);
    }
    if (max_allowed_ctx > 0 && prompt_tokens_count > max_allowed_ctx) {
        return Status::invalid_argument("Prompt token count (" + std::to_string(prompt_tokens_count) +
                                       ") exceeds context size (" + std::to_string(max_allowed_ctx) + ")");
    }
    uint32_t extra_tokens = (request.sampling.beam_size > 1 ? request.sampling.beam_size : 1) * request.sampling.max_output_tokens;
    uint32_t requested_ctx = prompt_tokens_count + extra_tokens;
    if (max_allowed_ctx > 0 && requested_ctx > max_allowed_ctx) {
        requested_ctx = max_allowed_ctx;
    }
    cparams.n_ctx = std::max<uint32_t>(512, requested_ctx);
    if (max_allowed_ctx > 0 && cparams.n_ctx > max_allowed_ctx) {
        cparams.n_ctx = max_allowed_ctx;
    }

    uint32_t max_token_cnt = model.config.decoder_max_token_count > 0 ? model.config.decoder_max_token_count : 512;
    if (request.sampling.beam_size > 1 && max_token_cnt < static_cast<uint32_t>(request.sampling.beam_size)) {
        max_token_cnt = request.sampling.beam_size;
    }
    cparams.n_batch = std::min<uint32_t>(max_token_cnt, cparams.n_ctx);
    uint32_t configured_ubatch = model.config.parsed_run_config.model.n_ubatch > 0
                                     ? static_cast<uint32_t>(model.config.parsed_run_config.model.n_ubatch)
                                     : cparams.n_batch;
    cparams.n_ubatch = std::min(configured_ubatch, cparams.n_batch);
    cparams.n_seq_max = (request.sampling.beam_size > 1) ? request.sampling.beam_size : 1;
    cparams.embeddings = (request.output.verbose_flags & INFERLET_VERBOSE_F_OUTPUT_EMBEDDING) != 0;

    int threads = 4;
    if (model.config.parsed_run_config.model.threads > 0) {
        threads = model.config.parsed_run_config.model.threads;
    } else {
        unsigned int hw = std::thread::hardware_concurrency();
        if (hw > 0) threads = std::min<int>(8, std::max<int>(1, hw));
    }
    cparams.n_threads = threads;
    cparams.n_threads_batch = (model.config.parsed_run_config.model.threads_batch > 0) ?
                              model.config.parsed_run_config.model.threads_batch : threads;

    log_debug("Creating llama context: n_ctx=" + std::to_string(cparams.n_ctx) +
              ", n_batch=" + std::to_string(cparams.n_batch) +
              ", threads=" + std::to_string(threads) +
              ", embeddings=" + std::to_string(cparams.embeddings));

    llama_context *ctx = llama_init_from_model(model.model.get(), cparams);
    if (!ctx) {
        return Status::out_of_memory("Failed to allocate llama_context (n_ctx=" + std::to_string(cparams.n_ctx) + ")");
    }

    if (request.lora_id > 0) {
        if (!model.lora_registry || !model.lora_registry->has_adapter(request.lora_id)) {
            llama_free(ctx);
            return Status::unsupported("LoRA adapter ID " + std::to_string(request.lora_id) + " not registered on model");
        }
        llama_adapter_lora *adapter = model.lora_registry->get_adapter(request.lora_id);
        Status lora_st = set_context_lora(ctx, adapter, 1.0f);
        if (!lora_st) {
            llama_free(ctx);
            return lora_st;
        }
    }

    uint32_t actual_n_ctx = llama_n_ctx(ctx);
    log_debug("Created llama context: actual n_ctx=" + std::to_string(actual_n_ctx));

    out_ctx.reset(ctx);
    return Status::ok();
}

uint32_t LlamaEngine::get_context_n_ctx(const llama_context *ctx) {
    if (!ctx) return 0;
    return llama_n_ctx(ctx);
}

void LlamaEngine::set_abort_callback(llama_context *ctx, const std::atomic<bool> *cancel_flag) {
    if (!ctx) return;
    if (cancel_flag) {
        llama_set_abort_callback(ctx, [](void *data) -> bool {
            auto *flag = static_cast<const std::atomic<bool> *>(data);
            return flag && flag->load();
        }, const_cast<void *>(static_cast<const void *>(cancel_flag)));
    } else {
        llama_set_abort_callback(ctx, nullptr, nullptr);
    }
}

Status LlamaEngine::decode_prompt(llama_context *ctx,
                                  const int *tokens,
                                  int count,
                                  int batch_size,
                                  const std::atomic<bool> *cancel_flag) {
    if (!ctx) return Status::invalid_handle("Context is NULL");
    if (!tokens || count <= 0) return Status::invalid_argument("Invalid prompt tokens");
    if (batch_size <= 0) batch_size = 512;

    llama_batch batch = llama_batch_init(batch_size, 0, 1);

    int processed = 0;
    while (processed < count) {
        if (cancel_flag && cancel_flag->load()) {
            llama_batch_free(batch);
            return Status::cancelled("Task cancelled during prompt decode");
        }

        int chunk = std::min(batch_size, count - processed);
        batch.n_tokens = chunk;
        for (int i = 0; i < chunk; ++i) {
            batch.token[i] = tokens[processed + i];
            batch.pos[i] = processed + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = (processed + i == count - 1) ? 1 : 0;
        }

        int32_t ret = llama_decode(ctx, batch);
        if (ret != 0) {
            llama_batch_free(batch);
            return Status::inference_error("llama_decode failed during prompt decode with code " + std::to_string(ret));
        }

        processed += chunk;
    }

    llama_batch_free(batch);
    return Status::ok();
}

Status LlamaEngine::decode_token(llama_context *ctx,
                                 int token,
                                 int pos) {
    if (!ctx) return Status::invalid_handle("Context is NULL");

    llama_batch batch = llama_batch_init(1, 0, 1);
    batch.n_tokens = 1;
    batch.token[0] = token;
    batch.pos[0] = pos;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0] = 1;

    int32_t ret = llama_decode(ctx, batch);
    llama_batch_free(batch);

    if (ret != 0) {
        return Status::inference_error("llama_decode failed during token decode with code " + std::to_string(ret));
    }
    return Status::ok();
}

Status LlamaEngine::decode_token_batch(llama_context *ctx,
                                       const std::vector<DecodeTokenBatchItem> &items) {
    if (!ctx) return Status::invalid_handle("Context is NULL");
    if (items.empty()) return Status::ok();

    llama_batch batch = llama_batch_init(static_cast<int32_t>(items.size()), 0, 1);
    batch.n_tokens = static_cast<int32_t>(items.size());
    for (size_t i = 0; i < items.size(); ++i) {
        batch.token[i] = items[i].token;
        batch.pos[i] = items[i].pos;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = items[i].seq_id;
        batch.logits[i] = items[i].output_logits ? 1 : 0;
    }

    int32_t ret = llama_decode(ctx, batch);
    llama_batch_free(batch);

    if (ret != 0) {
        return Status::inference_error("llama_decode failed during batch decode with code " + std::to_string(ret));
    }
    return Status::ok();
}

float *LlamaEngine::get_logits(llama_context *ctx) {
    if (!ctx) return nullptr;
    return llama_get_logits_ith(ctx, -1);
}

float *LlamaEngine::get_logits_ith(llama_context *ctx, int i) {
    if (!ctx) return nullptr;
    return llama_get_logits_ith(ctx, i);
}

float *LlamaEngine::get_embeddings(llama_context *ctx) {
    if (!ctx) return nullptr;
    return llama_get_embeddings_ith(ctx, -1);
}

int LlamaEngine::get_model_n_embd(const llama_model *model) {
    if (!model) return 0;
    return llama_model_n_embd(model);
}

void LlamaEngine::seq_copy(llama_context *ctx, int seq_src, int seq_dst, int p0, int p1) {
    if (!ctx) return;
    llama_memory_seq_cp(llama_get_memory(ctx), seq_src, seq_dst, p0, p1);
}

void LlamaEngine::seq_remove(llama_context *ctx, int seq_id, int p0, int p1) {
    if (!ctx) return;
    llama_memory_seq_rm(llama_get_memory(ctx), seq_id, p0, p1);
}

void LlamaEngine::seq_keep(llama_context *ctx, int seq_id) {
    if (!ctx) return;
    llama_memory_seq_keep(llama_get_memory(ctx), seq_id);
}

void LlamaEngine::memory_clear(llama_context *ctx, bool data) {
    if (!ctx) return;
    llama_memory_clear(llama_get_memory(ctx), data);
}

int LlamaEngine::seq_pos_max(llama_context *ctx, int seq_id) {
    if (!ctx) return -1;
    return llama_memory_seq_pos_max(llama_get_memory(ctx), seq_id);
}

bool LlamaEngine::is_kv_empty(llama_context *ctx, int n_seq_max) {
    if (!ctx) return true;
    for (int s = 0; s < n_seq_max; ++s) {
        if (llama_memory_seq_pos_max(llama_get_memory(ctx), s) != -1) {
            return false;
        }
    }
    return true;
}

LlamaSamplerPtr LlamaEngine::clone_grammar_sampler(const llama_sampler *smpl) {
    if (!smpl) return nullptr;
    return LlamaSamplerPtr(llama_sampler_clone(smpl));
}

Status LlamaEngine::create_grammar_sampler(const llama_vocab *vocab,
                                           const std::string &grammar_str,
                                           const std::string &grammar_root,
                                           LlamaSamplerPtr &out_sampler) {
    if (!vocab) {
        return Status::invalid_handle("Vocab is NULL for grammar sampler");
    }
    llama_sampler *smpl = llama_sampler_init_grammar(vocab, grammar_str.c_str(),
                                                     grammar_root.empty() ? "root" : grammar_root.c_str());
    if (!smpl) {
        return Status::internal("Failed to compile GBNF grammar into llama_sampler: " + grammar_str);
    }
    out_sampler.reset(smpl);
    return Status::ok();
}

void LlamaEngine::apply_grammar(llama_sampler *grammar_sampler, float *logits, int vocab_size) {
    if (!grammar_sampler || !logits || vocab_size <= 0) return;
    std::vector<llama_token_data> cur_p_data(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        cur_p_data[i] = { i, logits[i], 0.0f };
    }
    llama_token_data_array cur_p = {
        cur_p_data.data(),
        static_cast<size_t>(vocab_size),
        -1,
        false
    };
    llama_sampler_apply(grammar_sampler, &cur_p);
    for (int i = 0; i < vocab_size; ++i) {
        logits[i] = cur_p.data[i].logit;
    }
}

void LlamaEngine::accept_grammar(llama_sampler *grammar_sampler, int token) {
    if (grammar_sampler && token >= 0) {
        llama_sampler_accept(grammar_sampler, token);
    }
}

void LlamaEngine::reset_grammar(llama_sampler *grammar_sampler) {
    if (grammar_sampler) {
        llama_sampler_reset(grammar_sampler);
    }
}

Status LlamaEngine::set_context_lora(llama_context *ctx, llama_adapter_lora *adapter, float scale) {
    if (!ctx) {
        return Status::invalid_handle("Context is NULL in set_context_lora");
    }
    if (adapter) {
        llama_adapter_lora *adapters[1] = { adapter };
        float scales[1] = { scale };
        int32_t res = llama_set_adapters_lora(ctx, adapters, 1, scales);
        if (res != 0) {
            return Status::internal("llama_set_adapters_lora failed with code: " + std::to_string(res));
        }
    } else {
        int32_t res = llama_set_adapters_lora(ctx, nullptr, 0, nullptr);
        if (res != 0) {
            return Status::internal("Failed to clear LoRA adapters on context: " + std::to_string(res));
        }
    }
    return Status::ok();
}

Status LlamaEngine::get_model_capabilities(const llama_model *model, ModelCapabilities &out_caps) {
    if (!model) {
        return Status::invalid_handle("Model is NULL in get_model_capabilities");
    }
    char buf[256];
    buf[0] = '\0';
    int32_t len = llama_model_meta_val_str(model, "general.architecture", buf, sizeof(buf));
    if (len > 0) {
        out_caps.architecture = std::string(buf);
    }
    buf[0] = '\0';
    len = llama_model_meta_val_str(model, "general.name", buf, sizeof(buf));
    if (len > 0) {
        out_caps.model_name = std::string(buf);
    }
    if (out_caps.architecture.empty() || out_caps.model_name.empty()) {
        buf[0] = '\0';
        llama_model_desc(model, buf, sizeof(buf));
        if (out_caps.model_name.empty()) {
            out_caps.model_name = std::string(buf);
        }
    }

    auto to_lower = [](std::string s) {
        for (char &c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        return s;
    };
    std::string arch_lower = to_lower(out_caps.architecture);
    std::string name_lower = to_lower(out_caps.model_name);

    out_caps.supports_thinking = (arch_lower.find("qwen3") != std::string::npos ||
                                  name_lower.find("qwen3") != std::string::npos ||
                                  name_lower.find("qwen-3") != std::string::npos);

    const char *tmpl = llama_model_chat_template(model, nullptr);
    out_caps.supports_tools = (tmpl != nullptr && std::strlen(tmpl) > 0);
    out_caps.supports_lora = true;
    out_caps.supports_vision = false;
    return Status::ok();
}

} // namespace inferlet

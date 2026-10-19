#ifndef INFERLET_ENGINE_LLAMA_ENGINE_H_INCLUDED
#define INFERLET_ENGINE_LLAMA_ENGINE_H_INCLUDED

#include "support/status.h"
#include "core/model_config.h"
#include <memory>
#include <string>
#include <atomic>

struct llama_model;
struct llama_context;
struct llama_vocab;
struct llama_adapter_lora;
struct llama_sampler;

namespace inferlet {

struct SharedModel;
struct TaskRequest;
struct ModelCapabilities;

struct LlamaModelDeleter {
    void operator()(llama_model *m) const;
};
using LlamaModelPtr = std::unique_ptr<llama_model, LlamaModelDeleter>;

struct LlamaContextDeleter {
    void operator()(llama_context *c) const;
};
using LlamaContextPtr = std::unique_ptr<llama_context, LlamaContextDeleter>;

struct LlamaLoraDeleter {
    void operator()(llama_adapter_lora *l) const;
};
using LlamaLoraPtr = std::unique_ptr<llama_adapter_lora, LlamaLoraDeleter>;

struct LlamaSamplerDeleter {
    void operator()(llama_sampler *s) const;
};
using LlamaSamplerPtr = std::unique_ptr<llama_sampler, LlamaSamplerDeleter>;

void global_llama_backend_init();
void global_llama_backend_free();

class LlamaEngine {
public:
    static Status load_model(const ModelConfig &config,
                             const std::string &model_path,
                             LlamaModelPtr &out_model,
                             const llama_vocab *&out_vocab);

    static Status load_lora(llama_model *model,
                            const std::string &lora_path,
                            LlamaLoraPtr &out_lora);

    static const char *get_model_chat_template(const llama_model *model, const char *name = nullptr);

    static int32_t apply_chat_template(const char *tmpl,
                                       const std::vector<std::pair<std::string, std::string>> &chat,
                                       bool add_ass,
                                       std::string &out_formatted);

    static int get_vocab_bos(const llama_vocab *vocab);
    static int get_vocab_eos(const llama_vocab *vocab);
    static int32_t get_vocab_n_tokens(const llama_vocab *vocab);
    static bool is_vocab_eog(const llama_vocab *vocab, int token);

    static Status create_context(const struct SharedModel &model,
                                 const struct TaskRequest &request,
                                 uint32_t prompt_tokens_count,
                                 LlamaContextPtr &out_ctx);

    static Status decode_prompt(llama_context *ctx,
                                const int *tokens,
                                int count,
                                int batch_size,
                                const std::atomic<bool> *cancel_flag = nullptr);

    static Status decode_token(llama_context *ctx,
                               int token,
                               int pos);

    static uint32_t get_context_n_ctx(const llama_context *ctx);
    static void set_abort_callback(llama_context *ctx, const std::atomic<bool> *cancel_flag);

    static float *get_logits(llama_context *ctx);
    static float *get_logits_ith(llama_context *ctx, int i);
    static float *get_embeddings(llama_context *ctx);
    static int get_model_n_embd(const llama_model *model);

    struct DecodeTokenBatchItem {
        int token;
        int pos;
        int seq_id;
        bool output_logits;
    };

    static Status decode_token_batch(llama_context *ctx,
                                     const std::vector<DecodeTokenBatchItem> &items);

    static void seq_copy(llama_context *ctx, int seq_src, int seq_dst, int p0 = -1, int p1 = -1);
    static void seq_remove(llama_context *ctx, int seq_id, int p0 = -1, int p1 = -1);
    static void seq_keep(llama_context *ctx, int seq_id);
    static void memory_clear(llama_context *ctx, bool data = true);
    static int seq_pos_max(llama_context *ctx, int seq_id);
    static bool is_kv_empty(llama_context *ctx, int n_seq_max);

    static Status create_grammar_sampler(const llama_vocab *vocab,
                                         const std::string &grammar_str,
                                         const std::string &grammar_root,
                                         LlamaSamplerPtr &out_sampler);
    static LlamaSamplerPtr clone_grammar_sampler(const llama_sampler *smpl);
    static void apply_grammar(llama_sampler *grammar_sampler, float *logits, int vocab_size);
    static void accept_grammar(llama_sampler *grammar_sampler, int token);
    static void reset_grammar(llama_sampler *grammar_sampler);

    static Status set_context_lora(llama_context *ctx, llama_adapter_lora *adapter, float scale = 1.0f);
    static Status get_model_capabilities(const llama_model *model, ModelCapabilities &out_caps);
};

} // namespace inferlet

#endif // INFERLET_ENGINE_LLAMA_ENGINE_H_INCLUDED

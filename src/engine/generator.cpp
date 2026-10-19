#include "generator.h"
#include "beam_search.h"
#include "prompt_builder.h"
#include "sampler.h"
#include "guided_choice.h"
#include "token_codec.h"
#include "llama_engine.h"
#include "support/logging.h"
#include "multimodal/mtmd_backend.h"

namespace inferlet {

Status Generator::run(const TaskRequest &request,
                      const std::shared_ptr<SharedModel> &model,
                      ResultSink &sink,
                      const std::atomic<bool> *cancel_flag) {
    if (!model) {
        return Status::invalid_handle("Model is NULL");
    }
    {
        std::unique_lock<std::mutex> lock(model->mutex);
        if (model->load_state == LoadState::Loading) {
            model->load_cv.wait(lock, [&model]() {
                return model->load_state != LoadState::Loading;
            });
        }
        if (model->load_state == LoadState::Failed) {
            return model->load_status.is_ok() ? Status::model_load() : model->load_status;
        }
    }
    if (!model->model || !model->vocab) {
        return Status::invalid_handle("Model is not loaded or invalid");
    }

    if (request.sampling.beam_size > 1) {
        return BeamSearchExecutor::run(request, model, sink, cancel_flag);
    }

    PreparedPrompt prep;
    Status st = PromptBuilder::build(request, model, prep);
    if (!st) {
        sink.on_finish(st);
        return st;
    }

    st = sink.on_input(prep.verbose_input);
    if (!st) {
        sink.on_finish(st);
        return st;
    }

    size_t max_verbose_budget = model->config.parsed_run_config.limits.max_verbose_bytes;
    if (max_verbose_budget == 0) {
        max_verbose_budget = 536870912; // 512MB default
    }

    int vocab_size = LlamaEngine::get_vocab_n_tokens(model->vocab);
    int n_embd = (request.output.verbose_flags & INFERLET_VERBOSE_F_OUTPUT_EMBEDDING) ?
                 LlamaEngine::get_model_n_embd(model->model.get()) : 0;
    int max_tokens = request.sampling.max_output_tokens;

    uint64_t total_estimated_verbose = 0;
    if (request.output.verbose_flags & INFERLET_VERBOSE_F_OUTPUT_LOGITS) {
        if (max_tokens > 0 && vocab_size > 0) {
            uint64_t logits_bytes = static_cast<uint64_t>(max_tokens) * static_cast<uint64_t>(vocab_size) * sizeof(float);
            if (logits_bytes > max_verbose_budget || total_estimated_verbose + logits_bytes > max_verbose_budget) {
                Status err = Status::invalid_argument("Estimated verbose logits memory (" + std::to_string(logits_bytes) +
                                                      " bytes) exceeds max_verbose_bytes (" + std::to_string(max_verbose_budget) + ")");
                sink.on_finish(err);
                return err;
            }
            total_estimated_verbose += logits_bytes;
        }
    }

    if (request.output.verbose_flags & INFERLET_VERBOSE_F_OUTPUT_EMBEDDING) {
        if (max_tokens > 0 && n_embd > 0) {
            uint64_t embd_bytes = static_cast<uint64_t>(max_tokens) * static_cast<uint64_t>(n_embd) * sizeof(float);
            if (embd_bytes > max_verbose_budget || total_estimated_verbose + embd_bytes > max_verbose_budget) {
                Status err = Status::invalid_argument("Estimated verbose embedding memory (" + std::to_string(embd_bytes) +
                                                      " bytes) exceeds max_verbose_bytes (" + std::to_string(max_verbose_budget) + ")");
                sink.on_finish(err);
                return err;
            }
            total_estimated_verbose += embd_bytes;
        }
    }

    LlamaContextPtr ctx;
    st = LlamaEngine::create_context(*model, request, static_cast<uint32_t>(prep.tokens.size()), ctx);
    if (!st) {
        sink.on_finish(st);
        return st;
    }

    LlamaEngine::set_abort_callback(ctx.get(), cancel_flag);
    uint32_t n_ctx = LlamaEngine::get_context_n_ctx(ctx.get());
    if (prep.tokens.size() >= n_ctx) {
        Status err = Status::invalid_argument("Prompt token count (" + std::to_string(prep.tokens.size()) +
                                              ") exceeds context size (" + std::to_string(n_ctx) + ")");
        sink.on_finish(err);
        return err;
    }

    int next_position = static_cast<int>(prep.tokens.size());
    if (prep.multimodal) {
        st = model->multimodal->prefill(*prep.multimodal, ctx.get(), cancel_flag, next_position);
    } else {
        st = LlamaEngine::decode_prompt(ctx.get(), prep.tokens.data(),
                                        static_cast<int>(prep.tokens.size()),
                                        model->config.decoder_max_token_count, cancel_flag);
    }
    if (!st) {
        sink.on_finish(st);
        return st;
    }

    LlamaSamplerPtr grammar_sampler;
    if (!request.guided_choices.empty()) {
        std::string grammar_str;
        Status st_g = GuidedChoice::build_grammar(request.guided_choices, grammar_str);
        if (!st_g) {
            sink.on_finish(st_g);
            return st_g;
        }
        st_g = LlamaEngine::create_grammar_sampler(model->vocab, grammar_str, "root", grammar_sampler);
        if (!st_g) {
            sink.on_finish(st_g);
            return st_g;
        }
    }

    Sampler sampler(request.sampling, model->vocab, grammar_sampler.get());
    for (int t : prep.tokens) {
        if (t >= 0) sampler.accept_prompt_token(t);
    }

    std::string accumulated_text;
    for (int step = 0; step < max_tokens; ++step) {
        if (cancel_flag && cancel_flag->load()) {
            sink.on_finish(Status::cancelled());
            return Status::cancelled();
        }

        float *raw_logits = LlamaEngine::get_logits(ctx.get());
        if (!raw_logits) {
            Status err = Status::inference_error("Failed to retrieve logits from context");
            sink.on_finish(err);
            return err;
        }

        SampleResult sres;
        st = sampler.sample(raw_logits, vocab_size, sres);
        if (!st) {
            sink.on_finish(st);
            return st;
        }

        int selected = sres.token;
        sampler.accept(selected);

        if (sampler.is_eog(selected)) {
            log_debug("EOG reached at step " + std::to_string(step));
            break;
        }

        int pos = next_position + step;
        if (static_cast<uint32_t>(pos) >= n_ctx) {
            log_warn("Context size exceeded during generation at pos " + std::to_string(pos));
            Status err = Status::inference_error("Context size exceeded during generation");
            sink.on_finish(err);
            return err;
        }

        TokenStep step_data;
        step_data.token = selected;
        step_data.has_probability = (request.sampling.probability_type != 0);
        step_data.probability = sres.prob;

        if (request.output.verbose_flags & INFERLET_VERBOSE_F_OUTPUT_LOGITS) {
            step_data.logits.assign(raw_logits, raw_logits + vocab_size);
        }

        st = LlamaEngine::decode_token(ctx.get(), selected, pos);
        if (!st) {
            sink.on_finish(st);
            return st;
        }

        std::string piece = TokenCodec::token_to_piece(model->vocab, selected, request.output.keep_special_tokens);
        int offset = static_cast<int>(accumulated_text.size());
        accumulated_text.append(piece);

        step_data.piece = std::move(piece);
        step_data.offset = offset;

        if ((request.output.verbose_flags & INFERLET_VERBOSE_F_OUTPUT_EMBEDDING) && n_embd > 0) {
            float *embd = LlamaEngine::get_embeddings(ctx.get());
            if (embd) {
                step_data.embedding.assign(embd, embd + n_embd);
            }
        }

        st = sink.on_token(std::move(step_data));
        if (!st) {
            sink.on_finish(st);
            return st;
        }
    }

    sink.on_finish(Status::ok());
    return Status::ok();
}

} // namespace inferlet

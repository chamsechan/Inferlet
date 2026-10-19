#include "beam_search.h"
#include "prompt_builder.h"
#include "sampler.h"
#include "guided_choice.h"
#include "token_codec.h"
#include "llama_engine.h"
#include "support/logging.h"
#include <cmath>
#include <algorithm>
#include <limits>
#include <cassert>

namespace inferlet {

void BeamSearchExecutor::compute_log_softmax(const float *logits, int vocab_size, std::vector<double> &out_log_probs) {
    out_log_probs.resize(vocab_size);
    if (!logits || vocab_size <= 0) {
        return;
    }

    float max_logit = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < vocab_size; ++i) {
        if (logits[i] > -1e30f && logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }

    if (max_logit == -std::numeric_limits<float>::infinity()) {
        std::fill(out_log_probs.begin(), out_log_probs.end(), -std::numeric_limits<double>::infinity());
        return;
    }

    double sum_exp = 0.0;
    for (int i = 0; i < vocab_size; ++i) {
        if (logits[i] > -1e30f) {
            sum_exp += std::exp(static_cast<double>(logits[i] - max_logit));
        }
    }

    double log_sum_exp = std::log(sum_exp);
    for (int i = 0; i < vocab_size; ++i) {
        if (logits[i] > -1e30f) {
            out_log_probs[i] = static_cast<double>(logits[i] - max_logit) - log_sum_exp;
        } else {
            out_log_probs[i] = -std::numeric_limits<double>::infinity();
        }
    }
}

double BeamSearchExecutor::calculate_length_penalty(int length, float penalty_factor) {
    if (length <= 1 || penalty_factor == 0.0f) {
        return 1.0;
    }
    return std::pow(static_cast<double>(length), static_cast<double>(penalty_factor));
}

int BeamSearchExecutor::select_best_beam(const std::vector<BeamState> &beams, float length_penalty) {
    if (beams.empty()) return -1;
    int best_idx = 0;
    double best_penalized_score = -std::numeric_limits<double>::infinity();

    for (size_t i = 0; i < beams.size(); ++i) {
        int len = std::max(1, static_cast<int>(beams[i].tokens.size()));
        double penalty = calculate_length_penalty(len, length_penalty);
        double penalized_score = beams[i].cumulative_log_prob / penalty;
        if (penalized_score > best_penalized_score) {
            best_penalized_score = penalized_score;
            best_idx = static_cast<int>(i);
        }
    }
    return best_idx;
}

void BeamSearchExecutor::step_candidates(const std::vector<BeamState> &current_beams,
                                        const std::vector<std::vector<double>> &beam_log_probs,
                                        int vocab_size,
                                        int beam_size,
                                        const std::function<bool(int)> &is_eog_fn,
                                        std::vector<BeamCandidate> &out_top_candidates) {
    out_top_candidates.clear();
    std::vector<BeamCandidate> candidates;

    size_t active_log_prob_idx = 0;
    for (size_t b = 0; b < current_beams.size(); ++b) {
        if (current_beams[b].finished) {
            // Already finished beam is preserved as candidate without expanding
            BeamCandidate cand;
            cand.parent_beam_idx = static_cast<int>(b);
            cand.token = -1;
            cand.token_log_prob = 0.0;
            cand.new_cumulative_score = current_beams[b].cumulative_log_prob;
            cand.is_finished_beam = true;
            candidates.push_back(cand);
        } else {
            if (active_log_prob_idx >= beam_log_probs.size()) continue;
            const auto &log_probs = beam_log_probs[active_log_prob_idx++];
            for (int t = 0; t < vocab_size; ++t) {
                if (log_probs[t] <= -1e30) continue;
                BeamCandidate cand;
                cand.parent_beam_idx = static_cast<int>(b);
                cand.token = t;
                cand.token_log_prob = log_probs[t];
                cand.new_cumulative_score = current_beams[b].cumulative_log_prob + log_probs[t];
                cand.is_finished_beam = false;
                candidates.push_back(cand);
            }
        }
    }

    // Sort candidates descending by score, tie-breaking by parent_beam_idx then token
    std::sort(candidates.begin(), candidates.end(), [](const BeamCandidate &c1, const BeamCandidate &c2) {
        if (std::abs(c1.new_cumulative_score - c2.new_cumulative_score) > 1e-9) {
            return c1.new_cumulative_score > c2.new_cumulative_score;
        }
        if (c1.parent_beam_idx != c2.parent_beam_idx) {
            return c1.parent_beam_idx < c2.parent_beam_idx;
        }
        return c1.token < c2.token;
    });

    size_t count = std::min<size_t>(beam_size, candidates.size());
    out_top_candidates.assign(candidates.begin(), candidates.begin() + count);
}

Status BeamSearchExecutor::run(const TaskRequest &request,
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

    int beam_size = request.sampling.beam_size;
    if (beam_size <= 1) {
        return Status::invalid_argument("BeamSearchExecutor requires beam_size > 1");
    }

    constexpr int unsupported_verbose = INFERLET_VERBOSE_F_OUTPUT_LOGITS |
                                        INFERLET_VERBOSE_F_OUTPUT_EMBEDDING;
    if ((request.output.verbose_flags & unsupported_verbose) != 0) {
        Status err = Status::unsupported("Beam search does not support output logits or output embeddings");
        sink.on_finish(err);
        return err;
    }

    // Check random sampling conflict
    if (request.sampling.custom_temperature && request.sampling.temperature > 0.0f) {
        Status err = Status::invalid_argument("Beam search cannot be combined with temperature > 0");
        sink.on_finish(err);
        return err;
    }
    if (request.sampling.custom_top_p && request.sampling.top_p < 1.0f) {
        Status err = Status::invalid_argument("Beam search cannot be combined with top_p < 1.0");
        sink.on_finish(err);
        return err;
    }
    if (request.sampling.custom_top_k && request.sampling.top_k > 0) {
        Status err = Status::invalid_argument("Beam search cannot be combined with top_k");
        sink.on_finish(err);
        return err;
    }

    log_info("BeamSearchExecutor (beam_size=" + std::to_string(beam_size) +
             "): using deterministic repetition + grammar + log-softmax");

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

    int vocab_size = LlamaEngine::get_vocab_n_tokens(model->vocab);
    int max_tokens = request.sampling.max_output_tokens;

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

    // Step 1: Prompt decode once on sequence 0
    st = LlamaEngine::decode_prompt(ctx.get(), prep.tokens.data(),
                                    static_cast<int>(prep.tokens.size()),
                                    model->config.decoder_max_token_count,
                                    cancel_flag);
    if (!st) {
        LlamaEngine::seq_remove(ctx.get(), -1, -1, -1);
        sink.on_finish(st);
        return st;
    }

    if (cancel_flag && cancel_flag->load()) {
        LlamaEngine::seq_remove(ctx.get(), -1, -1, -1);
        sink.on_finish(Status::cancelled());
        return Status::cancelled();
    }

    // Grammar setup if guided_choices is non-empty
    LlamaSamplerPtr root_grammar_sampler;
    if (!request.guided_choices.empty()) {
        std::string grammar_str;
        Status st_g = GuidedChoice::build_grammar(request.guided_choices, grammar_str);
        if (!st_g) {
            LlamaEngine::seq_remove(ctx.get(), -1, -1, -1);
            sink.on_finish(st_g);
            return st_g;
        }
        st_g = LlamaEngine::create_grammar_sampler(model->vocab, grammar_str, "root", root_grammar_sampler);
        if (!st_g) {
            LlamaEngine::seq_remove(ctx.get(), -1, -1, -1);
            sink.on_finish(st_g);
            return st_g;
        }
    }

    // Repetition history for prompt tokens
    std::vector<int> prompt_history;
    int max_hist = (request.sampling.repeat_last_n > 0) ? request.sampling.repeat_last_n : 64;
    for (int t : prep.tokens) {
        prompt_history.push_back(t);
        if (static_cast<int>(prompt_history.size()) > max_hist) {
            prompt_history.erase(prompt_history.begin());
        }
    }

    // Step 2 & 3: Initial candidate expansion and sequence allocation
    float *prompt_raw_logits = LlamaEngine::get_logits(ctx.get());
    if (!prompt_raw_logits) {
        LlamaEngine::seq_remove(ctx.get(), -1, -1, -1);
        Status err = Status::inference_error("Failed to retrieve prompt logits");
        sink.on_finish(err);
        return err;
    }

    std::vector<float> prompt_logits(prompt_raw_logits, prompt_raw_logits + vocab_size);
    Sampler::apply_repetition_penalty(prompt_logits.data(), vocab_size, prompt_history, request.sampling.repetition_penalty);
    if (root_grammar_sampler) {
        LlamaEngine::apply_grammar(root_grammar_sampler.get(), prompt_logits.data(), vocab_size);
    }

    std::vector<double> prompt_log_probs;
    compute_log_softmax(prompt_logits.data(), vocab_size, prompt_log_probs);

    std::vector<std::pair<int, double>> initial_ranked;
    initial_ranked.reserve(vocab_size);
    for (int t = 0; t < vocab_size; ++t) {
        if (prompt_log_probs[t] > -1e30) {
            initial_ranked.emplace_back(t, prompt_log_probs[t]);
        }
    }

    std::sort(initial_ranked.begin(), initial_ranked.end(), [](const auto &a, const auto &b) {
        if (std::abs(a.second - b.second) > 1e-9) {
            return a.second > b.second;
        }
        return a.first < b.first;
    });

    if (initial_ranked.empty()) {
        LlamaEngine::seq_remove(ctx.get(), -1, -1, -1);
        Status err = Status::inference_error("No valid candidate tokens produced from prompt");
        sink.on_finish(err);
        return err;
    }

    size_t actual_beams = std::min<size_t>(beam_size, initial_ranked.size());
    std::vector<BeamState> beams(actual_beams);

    for (size_t b = 0; b < actual_beams; ++b) {
        beams[b].seq_id = static_cast<int>(b);
        if (b > 0) {
            // Copy prompt KV sequence from seq 0 to seq b
            LlamaEngine::seq_copy(ctx.get(), 0, static_cast<int>(b), -1, -1);
        }

        int tok = initial_ranked[b].first;
        double lprob = initial_ranked[b].second;
        bool is_eog = LlamaEngine::is_vocab_eog(model->vocab, tok);

        beams[b].cumulative_log_prob = lprob;
        beams[b].repetition_history = prompt_history;

        if (is_eog) {
            beams[b].finished = true;
        } else {
            beams[b].finished = false;
            beams[b].tokens.push_back(tok);
            beams[b].token_probs.push_back(static_cast<float>(std::exp(lprob)));
            beams[b].repetition_history.push_back(tok);
            if (static_cast<int>(beams[b].repetition_history.size()) > max_hist) {
                beams[b].repetition_history.erase(beams[b].repetition_history.begin());
            }
        }

        if (root_grammar_sampler) {
            beams[b].grammar_sampler = LlamaEngine::clone_grammar_sampler(root_grammar_sampler.get());
            LlamaEngine::accept_grammar(beams[b].grammar_sampler.get(), tok);
        }
    }

    // Step 4 to 6: Generation loop
    float length_penalty = model->config.parsed_run_config.sampling.beam_length_penalty;
    auto is_eog_fn = [&model](int tok) -> bool {
        return LlamaEngine::is_vocab_eog(model->vocab, tok);
    };

    for (int step = 1; step < max_tokens; ++step) {
        if (cancel_flag && cancel_flag->load()) {
            LlamaEngine::seq_remove(ctx.get(), -1, -1, -1);
            sink.on_finish(Status::cancelled());
            return Status::cancelled();
        }

        bool all_finished = std::all_of(beams.begin(), beams.end(), [](const BeamState &b) {
            return b.finished;
        });
        if (all_finished) {
            break;
        }

        // Gather active beams for batched decoding
        std::vector<size_t> active_indices;
        std::vector<LlamaEngine::DecodeTokenBatchItem> batch_items;

        for (size_t b = 0; b < beams.size(); ++b) {
            if (!beams[b].finished) {
                active_indices.push_back(b);
                int pos = static_cast<int>(prep.tokens.size()) + static_cast<int>(beams[b].tokens.size()) - 1;
                if (static_cast<uint32_t>(pos) >= n_ctx) {
                    beams[b].finished = true;
                    continue;
                }
                LlamaEngine::DecodeTokenBatchItem item;
                item.token = beams[b].tokens.back();
                item.pos = pos;
                item.seq_id = beams[b].seq_id;
                item.output_logits = true;
                batch_items.push_back(item);
            }
        }

        if (batch_items.empty()) {
            break;
        }

        st = LlamaEngine::decode_token_batch(ctx.get(), batch_items);
        if (!st) {
            LlamaEngine::seq_remove(ctx.get(), -1, -1, -1);
            sink.on_finish(st);
            return st;
        }

        // Collect logits and compute log-softmax for each active beam
        std::vector<std::vector<double>> active_log_probs(batch_items.size());
        for (size_t i = 0; i < batch_items.size(); ++i) {
            size_t beam_idx = active_indices[i];
            float *raw_logits = LlamaEngine::get_logits_ith(ctx.get(), static_cast<int>(i));
            if (!raw_logits) {
                LlamaEngine::seq_remove(ctx.get(), -1, -1, -1);
                Status err = Status::inference_error("Failed to retrieve batch logits for beam " + std::to_string(beam_idx));
                sink.on_finish(err);
                return err;
            }

            std::vector<float> logits(raw_logits, raw_logits + vocab_size);
            Sampler::apply_repetition_penalty(logits.data(), vocab_size, beams[beam_idx].repetition_history, request.sampling.repetition_penalty);
            if (beams[beam_idx].grammar_sampler) {
                LlamaEngine::apply_grammar(beams[beam_idx].grammar_sampler.get(), logits.data(), vocab_size);
            }

            compute_log_softmax(logits.data(), vocab_size, active_log_probs[i]);
        }

        // Step 5: Expand and keep global top beams
        std::vector<BeamCandidate> top_candidates;
        step_candidates(beams, active_log_probs, vocab_size, beam_size, is_eog_fn, top_candidates);

        if (top_candidates.empty()) {
            break;
        }

        // KV cache sequence reallocation
        std::vector<int> parent_child_count(beams.size(), 0);
        for (const auto &cand : top_candidates) {
            parent_child_count[cand.parent_beam_idx]++;
        }

        std::vector<int> free_seq_ids;
        for (size_t p = 0; p < beams.size(); ++p) {
            if (parent_child_count[p] == 0) {
                int s = beams[p].seq_id;
                LlamaEngine::seq_remove(ctx.get(), s, -1, -1);
                free_seq_ids.push_back(s);
            }
        }

        std::vector<BeamState> new_beams(top_candidates.size());
        std::vector<int> parent_assigned_seq(beams.size(), -1);

        for (size_t i = 0; i < top_candidates.size(); ++i) {
            const auto &cand = top_candidates[i];
            int p = cand.parent_beam_idx;

            if (cand.is_finished_beam) {
                // Carried-forward finished beam
                new_beams[i] = std::move(beams[p]);
                parent_assigned_seq[p] = new_beams[i].seq_id;
                continue;
            }

            int child_seq = -1;
            if (parent_assigned_seq[p] == -1) {
                // First child inherits parent's sequence slot
                child_seq = beams[p].seq_id;
                parent_assigned_seq[p] = child_seq;
            } else {
                // Sibling child gets free sequence slot and copies parent KV cache
                if (!free_seq_ids.empty()) {
                    child_seq = free_seq_ids.back();
                    free_seq_ids.pop_back();
                } else {
                    child_seq = beams[p].seq_id;
                }
                LlamaEngine::seq_copy(ctx.get(), parent_assigned_seq[p], child_seq, -1, -1);
            }

            bool is_eog = is_eog_fn(cand.token);

            BeamState child;
            child.seq_id = child_seq;
            child.tokens = beams[p].tokens;
            child.token_probs = beams[p].token_probs;
            child.cumulative_log_prob = cand.new_cumulative_score;
            child.repetition_history = beams[p].repetition_history;

            if (is_eog) {
                child.finished = true;
            } else {
                child.finished = false;
                child.tokens.push_back(cand.token);
                child.token_probs.push_back(static_cast<float>(std::exp(cand.token_log_prob)));
                child.repetition_history.push_back(cand.token);
                if (static_cast<int>(child.repetition_history.size()) > max_hist) {
                    child.repetition_history.erase(
                        child.repetition_history.begin(),
                        child.repetition_history.begin() + (child.repetition_history.size() - max_hist));
                }
            }

            if (beams[p].grammar_sampler) {
                child.grammar_sampler = LlamaEngine::clone_grammar_sampler(beams[p].grammar_sampler.get());
                LlamaEngine::accept_grammar(child.grammar_sampler.get(), cand.token);
            }

            new_beams[i] = std::move(child);
        }

        beams = std::move(new_beams);
    }

    // Step 7: Select winning beam using length penalty
    int best_idx = select_best_beam(beams, length_penalty);
    if (best_idx < 0 || best_idx >= static_cast<int>(beams.size())) {
        best_idx = 0;
    }
    const BeamState &winner = beams[best_idx];

    // Step 8: Clean up all KV sequences - zero residual guarantee
    LlamaEngine::seq_remove(ctx.get(), -1, -1, -1);
    bool clean = LlamaEngine::is_kv_empty(ctx.get(), beam_size);
    if (!clean) {
        log_warn("KV cache sequence residual detected after cleanup; forcing full memory clear");
        LlamaEngine::memory_clear(ctx.get(), true);
    }

    // Emit winning beam tokens to ResultSink
    std::string accumulated_text;
    for (size_t i = 0; i < winner.tokens.size(); ++i) {
        int tok = winner.tokens[i];
        TokenStep step_data;
        step_data.token = tok;
        step_data.has_probability = (request.sampling.probability_type != 0);
        step_data.probability = (i < winner.token_probs.size()) ? winner.token_probs[i] : 0.0f;

        std::string piece = TokenCodec::token_to_piece(model->vocab, tok, request.output.keep_special_tokens);
        int offset = static_cast<int>(accumulated_text.size());
        accumulated_text.append(piece);

        step_data.piece = std::move(piece);
        step_data.offset = offset;

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

#include "sampler.h"
#include "engine/llama_engine.h"
#include "support/logging.h"
#include <cmath>
#include <algorithm>
#include <unordered_set>
#include <limits>

namespace inferlet {

Sampler::Sampler(const SamplingConfig &config, const llama_vocab *vocab, llama_sampler *grammar_sampler)
    : config_(config), vocab_(vocab), grammar_sampler_(grammar_sampler), rng_(config.seed != 0 ? config.seed : 1337) {
}

void Sampler::reset() {
    repetition_history_.clear();
    rng_.seed(config_.seed != 0 ? config_.seed : 1337);
    if (grammar_sampler_) {
        LlamaEngine::reset_grammar(grammar_sampler_);
    }
}

void Sampler::accept_prompt_token(int token) {
    repetition_history_.push_back(token);
    int max_hist = (config_.repeat_last_n > 0) ? config_.repeat_last_n : 64;
    if (static_cast<int>(repetition_history_.size()) > max_hist) {
        repetition_history_.erase(repetition_history_.begin(),
                                  repetition_history_.begin() + (repetition_history_.size() - max_hist));
    }
}

void Sampler::accept(int token) {
    accept_prompt_token(token);
    if (grammar_sampler_) {
        LlamaEngine::accept_grammar(grammar_sampler_, token);
    }
}

bool Sampler::is_eog(int token) const {
    if (vocab_) {
        return LlamaEngine::is_vocab_eog(vocab_, token);
    }
    return false;
}

void Sampler::apply_repetition_penalty(float *logits, int vocab_size,
                                      const std::vector<int> &history,
                                      float penalty) {
    if (!logits || vocab_size <= 0 || penalty <= 0.0f || penalty == 1.0f || history.empty()) {
        return;
    }

    std::unordered_set<int> unique_tokens(history.begin(), history.end());
    for (int t : unique_tokens) {
        if (t >= 0 && t < vocab_size) {
            if (logits[t] > 0.0f) {
                logits[t] /= penalty;
            } else {
                logits[t] *= penalty;
            }
        }
    }
}

void Sampler::apply_temperature(float *logits, int vocab_size, float temp) {
    if (!logits || vocab_size <= 0 || temp <= 0.0f || temp == 1.0f) {
        return;
    }
    float inv_temp = 1.0f / temp;
    for (int i = 0; i < vocab_size; ++i) {
        logits[i] *= inv_temp;
    }
}

void Sampler::apply_top_k(std::vector<std::pair<int, float>> &candidates, int k) {
    if (k <= 0 || static_cast<size_t>(k) >= candidates.size()) {
        return;
    }
    std::partial_sort(candidates.begin(),
                      candidates.begin() + k,
                      candidates.end(),
                      [](const auto &a, const auto &b) { return a.second > b.second; });
    candidates.resize(k);
}

Status Sampler::apply_top_p(std::vector<std::pair<int, float>> &candidates, float p) {
    if (p <= 0.0f || p >= 1.0f || candidates.size() <= 1) {
        return Status::ok();
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const auto &a, const auto &b) { return a.second > b.second; });

    double max_logit = candidates[0].second;
    double sum = 0.0;
    std::vector<double> exps(candidates.size());
    for (size_t i = 0; i < candidates.size(); ++i) {
        exps[i] = std::exp(static_cast<double>(candidates[i].second - max_logit));
        sum += exps[i];
    }
    if (sum <= 0.0 || std::isnan(sum) || std::isinf(sum)) {
        return Status::inference_error("Invalid softmax denominator in top-p calculation");
    }

    double cum_prob = 0.0;
    size_t keep_count = 1;
    for (size_t i = 0; i < candidates.size(); ++i) {
        cum_prob += exps[i] / sum;
        if (cum_prob >= p) {
            keep_count = i + 1;
            break;
        }
    }

    candidates.resize(keep_count);
    return Status::ok();
}

Status Sampler::stable_softmax(const std::vector<std::pair<int, float>> &candidates,
                              std::vector<double> &out_probs) {
    if (candidates.empty()) {
        return Status::inference_error("Cannot compute softmax on empty candidates");
    }

    double max_logit = -std::numeric_limits<double>::infinity();
    for (const auto &c : candidates) {
        if (c.second > max_logit) max_logit = c.second;
    }

    if (std::isinf(max_logit)) {
        return Status::inference_error("Max logit is infinity in softmax");
    }

    out_probs.resize(candidates.size());
    double sum = 0.0;
    for (size_t i = 0; i < candidates.size(); ++i) {
        out_probs[i] = std::exp(static_cast<double>(candidates[i].second - max_logit));
        sum += out_probs[i];
    }

    if (sum <= 0.0 || std::isnan(sum) || std::isinf(sum)) {
        return Status::inference_error("Softmax denominator sum is non-positive or NaN/Inf");
    }

    double inv_sum = 1.0 / sum;
    for (size_t i = 0; i < out_probs.size(); ++i) {
        out_probs[i] *= inv_sum;
    }

    return Status::ok();
}

float Sampler::calculate_prob_before(const float *raw_logits, int vocab_size, int selected_token) {
    if (!raw_logits || vocab_size <= 0 || selected_token < 0 || selected_token >= vocab_size) {
        return 0.0f;
    }

    float max_l = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < vocab_size; ++i) {
        if (!std::isnan(raw_logits[i]) && raw_logits[i] > max_l) {
            max_l = raw_logits[i];
        }
    }
    if (std::isinf(max_l)) return 0.0f;

    double sum = 0.0;
    for (int i = 0; i < vocab_size; ++i) {
        if (!std::isnan(raw_logits[i])) {
            sum += std::exp(static_cast<double>(raw_logits[i] - max_l));
        }
    }
    if (sum <= 0.0 || std::isnan(sum) || std::isinf(sum)) return 0.0f;

    double sel_val = std::exp(static_cast<double>(raw_logits[selected_token] - max_l));
    return static_cast<float>(sel_val / sum);
}

Status Sampler::sample(const float *raw_logits, int vocab_size, SampleResult &out_result) {
    if (!raw_logits || vocab_size <= 0) {
        return Status::invalid_argument("Invalid raw logits or vocab size");
    }

    std::vector<float> work_logits(raw_logits, raw_logits + vocab_size);

    // Step 0: Guided grammar/choice masks illegal tokens before repetition penalty and sampling
    if (grammar_sampler_) {
        LlamaEngine::apply_grammar(grammar_sampler_, work_logits.data(), vocab_size);
    }

    // Step 1: Repetition penalty
    if (config_.repetition_penalty > 0.0f && config_.repetition_penalty != 1.0f && !repetition_history_.empty()) {
        apply_repetition_penalty(work_logits.data(), vocab_size, repetition_history_, config_.repetition_penalty);
    }

    // Step 2: Temperature
    if (config_.temperature > 0.0f && config_.temperature != 1.0f) {
        apply_temperature(work_logits.data(), vocab_size, config_.temperature);
    }

    // Build candidates list
    std::vector<std::pair<int, float>> candidates;
    candidates.reserve(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        if (!std::isnan(work_logits[i]) && !std::isinf(work_logits[i])) {
            candidates.emplace_back(i, work_logits[i]);
        }
    }
    if (candidates.empty()) {
        return Status::inference_error("All candidate logits are invalid or filtered out");
    }

    // Step 3: Top-K
    if (config_.top_k > 0 && config_.top_k < static_cast<int>(candidates.size())) {
        apply_top_k(candidates, config_.top_k);
    }

    // Step 4: Top-P
    if (config_.top_p > 0.0f && config_.top_p < 1.0f && candidates.size() > 1) {
        Status st = apply_top_p(candidates, config_.top_p);
        if (!st) return st;
    }

    int selected_token = -1;
    float p_after = 0.0f;

    if (config_.temperature <= 0.0f) {
        auto max_it = std::max_element(candidates.begin(), candidates.end(),
                                       [](const auto &a, const auto &b) { return a.second < b.second; });
        selected_token = max_it->first;

        if (config_.probability_type == 2) {
            std::vector<double> probs;
            Status st = stable_softmax(candidates, probs);
            if (!st) return st;
            size_t idx = std::distance(candidates.begin(), max_it);
            p_after = static_cast<float>(probs[idx]);
        }
    } else {
        std::vector<double> probs;
        Status st = stable_softmax(candidates, probs);
        if (!st) return st;

        std::uniform_real_distribution<double> dist(0.0, 1.0);
        double u = dist(rng_);
        double cum = 0.0;
        size_t chosen_idx = candidates.size() - 1;
        for (size_t i = 0; i < probs.size(); ++i) {
            cum += probs[i];
            if (u <= cum) {
                chosen_idx = i;
                break;
            }
        }
        selected_token = candidates[chosen_idx].first;
        p_after = static_cast<float>(probs[chosen_idx]);
    }

    out_result.token = selected_token;
    if (config_.probability_type == 1) {
        out_result.prob = calculate_prob_before(raw_logits, vocab_size, selected_token);
    } else if (config_.probability_type == 2) {
        out_result.prob = p_after;
    } else {
        out_result.prob = 0.0f;
    }

    return Status::ok();
}

} // namespace inferlet

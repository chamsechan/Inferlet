#ifndef INFERLET_ENGINE_SAMPLER_H_INCLUDED
#define INFERLET_ENGINE_SAMPLER_H_INCLUDED

#include "core/task_input.h"
#include "support/status.h"
#include <vector>
#include <random>
#include <cstdint>

struct llama_vocab;
struct llama_sampler;

namespace inferlet {

struct SampleResult {
    int token = -1;
    float prob = 0.0f;
};

class Sampler {
public:
    explicit Sampler(const SamplingConfig &config,
                     const llama_vocab *vocab = nullptr,
                     llama_sampler *grammar_sampler = nullptr);

    void reset();

    Status sample(const float *raw_logits, int vocab_size, SampleResult &out_result);

    void accept(int token);
    void accept_prompt_token(int token);

    bool is_eog(int token) const;

    // Helper static functions for pipeline steps & testing
    static void apply_repetition_penalty(float *logits, int vocab_size,
                                         const std::vector<int> &history,
                                         float penalty);

    static void apply_temperature(float *logits, int vocab_size, float temp);

    static void apply_top_k(std::vector<std::pair<int, float>> &candidates, int k);

    static Status apply_top_p(std::vector<std::pair<int, float>> &candidates, float p);

    static Status stable_softmax(const std::vector<std::pair<int, float>> &candidates,
                                 std::vector<double> &out_probs);

    static float calculate_prob_before(const float *raw_logits, int vocab_size, int selected_token);

    const std::vector<int> &repetition_history() const { return repetition_history_; }

private:
    SamplingConfig config_;
    const llama_vocab *vocab_ = nullptr;
    llama_sampler *grammar_sampler_ = nullptr;
    std::mt19937_64 rng_;
    std::vector<int> repetition_history_;
};

} // namespace inferlet

#endif // INFERLET_ENGINE_SAMPLER_H_INCLUDED

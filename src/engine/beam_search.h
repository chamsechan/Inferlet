#ifndef INFERLET_ENGINE_BEAM_SEARCH_H_INCLUDED
#define INFERLET_ENGINE_BEAM_SEARCH_H_INCLUDED

#include "support/status.h"
#include "core/task_input.h"
#include "core/model_registry.h"
#include "result_sink.h"
#include "llama_engine.h"
#include <atomic>
#include <vector>
#include <string>
#include <functional>

namespace inferlet {

struct BeamState {
    int seq_id = -1;
    std::vector<int> tokens;
    std::vector<float> token_probs;
    double cumulative_log_prob = 0.0;
    bool finished = false;
    std::vector<int> repetition_history;
    LlamaSamplerPtr grammar_sampler;
};

struct BeamCandidate {
    int parent_beam_idx = -1;
    int token = -1;
    double token_log_prob = 0.0;
    double new_cumulative_score = 0.0;
    bool is_finished_beam = false;
};

class BeamSearchExecutor {
public:
    static Status run(const TaskRequest &request,
                      const std::shared_ptr<SharedModel> &model,
                      ResultSink &sink,
                      const std::atomic<bool> *cancel_flag = nullptr);

    // Pure algorithmic helpers for testing & core logic
    static void compute_log_softmax(const float *logits, int vocab_size, std::vector<double> &out_log_probs);

    static double calculate_length_penalty(int length, float penalty_factor);

    static int select_best_beam(const std::vector<BeamState> &beams, float length_penalty);

    // Pure candidate selection step (usable in deterministic toy tests)
    static void step_candidates(const std::vector<BeamState> &current_beams,
                                const std::vector<std::vector<double>> &beam_log_probs,
                                int vocab_size,
                                int beam_size,
                                const std::function<bool(int)> &is_eog_fn,
                                std::vector<BeamCandidate> &out_top_candidates);
};

} // namespace inferlet

#endif // INFERLET_ENGINE_BEAM_SEARCH_H_INCLUDED

#include "inferlet.h"
#include "engine/sampler.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace inferlet;

void test_sampler_greedy() {
    SamplingConfig cfg;
    cfg.temperature = 0.0f;
    cfg.probability_type = 1; // p_before

    Sampler sampler(cfg);
    float logits[] = {1.0f, 5.0f, 2.0f, 0.5f};
    int vocab_size = 4;

    SampleResult res;
    Status st = sampler.sample(logits, vocab_size, res);
    assert(st.is_ok());
    assert(res.token == 1); // Argmax is index 1

    // Verify analytical softmax for p_before
    double max_l = 5.0;
    double denom = std::exp(1.0 - max_l) + std::exp(5.0 - max_l) + std::exp(2.0 - max_l) + std::exp(0.5 - max_l);
    double expected_p1 = std::exp(5.0 - max_l) / denom;
    assert(std::fabs(res.prob - static_cast<float>(expected_p1)) < 1e-5f);

    // Test prob_type = 2 (p_after) without top-p/top-k truncation -> matches p_before
    cfg.probability_type = 2;
    cfg.top_p = 1.0f;
    cfg.top_k = 0;
    Sampler sampler2(cfg);
    st = sampler2.sample(logits, vocab_size, res);
    assert(st.is_ok());
    assert(res.token == 1);
    assert(std::fabs(res.prob - static_cast<float>(expected_p1)) < 1e-5f);

    // Test prob_type = 2 (p_after) with top_p = 0.95 (truncates to tokens 1 and 2)
    cfg.top_p = 0.95f;
    Sampler sampler3(cfg);
    st = sampler3.sample(logits, vocab_size, res);
    assert(st.is_ok());
    assert(res.token == 1);
    double expected_p_after = std::exp(5.0 - max_l) / (std::exp(5.0 - max_l) + std::exp(2.0 - max_l));
    assert(std::fabs(res.prob - static_cast<float>(expected_p_after)) < 1e-5f);

    std::cout << "[PASS] test_sampler_greedy\n";
}

void test_sampler_repetition_penalty() {
    SamplingConfig cfg;
    cfg.temperature = 0.0f; // Greedy to observe deterministic winner
    cfg.repetition_penalty = 2.0f;
    cfg.repeat_last_n = 10;

    Sampler sampler(cfg);
    float logits[] = {1.0f, 5.0f, 4.5f, 0.5f};
    int vocab_size = 4;

    // First sample: token 1 has 5.0, token 2 has 4.5 -> token 1 wins
    SampleResult res;
    Status st = sampler.sample(logits, vocab_size, res);
    assert(st.is_ok());
    assert(res.token == 1);

    // Accept token 1 into history
    sampler.accept(res.token);
    assert(sampler.repetition_history().size() == 1);
    assert(sampler.repetition_history()[0] == 1);

    // Second sample with penalty: token 1 logit becomes 5.0 / 2.0 = 2.5
    // Token 2 logit remains 4.5 -> token 2 wins!
    st = sampler.sample(logits, vocab_size, res);
    assert(st.is_ok());
    assert(res.token == 2);

    // Accept token 2, now history has [1, 2]
    sampler.accept(res.token);
    assert(sampler.repetition_history().size() == 2);

    std::cout << "[PASS] test_sampler_repetition_penalty\n";
}

void test_sampler_temperature() {
    std::vector<std::pair<int, float>> candidates = {
        {0, 2.0f},
        {1, 1.0f}
    };

    std::vector<double> probs_t1;
    Status st = Sampler::stable_softmax(candidates, probs_t1);
    assert(st.is_ok());
    assert(std::fabs((probs_t1[0] + probs_t1[1]) - 1.0) < 1e-6);

    // With lower temperature (temp = 0.5), delta doubles: (2-1)/0.5 = 2.0 vs 1.0
    // Probability of index 0 should increase
    std::vector<std::pair<int, float>> candidates_low = {
        {0, 2.0f / 0.5f},
        {1, 1.0f / 0.5f}
    };
    std::vector<double> probs_low;
    st = Sampler::stable_softmax(candidates_low, probs_low);
    assert(st.is_ok());
    assert(probs_low[0] > probs_t1[0]);

    // With higher temperature (temp = 2.0), delta halves: (2-1)/2.0 = 0.5
    // Distribution should flatten
    std::vector<std::pair<int, float>> candidates_high = {
        {0, 2.0f / 2.0f},
        {1, 1.0f / 2.0f}
    };
    std::vector<double> probs_high;
    st = Sampler::stable_softmax(candidates_high, probs_high);
    assert(st.is_ok());
    assert(probs_high[0] < probs_t1[0]);

    std::cout << "[PASS] test_sampler_temperature\n";
}

void test_sampler_top_k() {
    std::vector<std::pair<int, float>> candidates;
    for (int i = 0; i < 10; ++i) {
        candidates.emplace_back(i, static_cast<float>(i));
    }

    Sampler::apply_top_k(candidates, 3);
    assert(candidates.size() == 3);
    // Highest 3 are 9, 8, 7
    bool has_9 = false, has_8 = false, has_7 = false;
    for (const auto &c : candidates) {
        if (c.first == 9) has_9 = true;
        if (c.first == 8) has_8 = true;
        if (c.first == 7) has_7 = true;
    }
    assert(has_9 && has_8 && has_7);

    std::cout << "[PASS] test_sampler_top_k\n";
}

void test_sampler_top_p() {
    // 4 items with sharply decreasing logits
    std::vector<std::pair<int, float>> candidates = {
        {0, 10.0f},
        {1, 8.0f},
        {2, 1.0f},
        {3, 0.0f}
    };

    // Calculate probabilities: item 0 has ~88%, item 1 has ~12%
    // If top_p = 0.8, item 0 alone exceeds 0.8 -> keep 1
    std::vector<std::pair<int, float>> c1 = candidates;
    Status st = Sampler::apply_top_p(c1, 0.8f);
    assert(st.is_ok());
    assert(c1.size() == 1);
    assert(c1[0].first == 0);

    // If top_p = 0.95, items 0 + 1 (~99%) -> keep 2
    std::vector<std::pair<int, float>> c2 = candidates;
    st = Sampler::apply_top_p(c2, 0.95f);
    assert(st.is_ok());
    assert(c2.size() == 2);
    assert(c2[0].first == 0 && c2[1].first == 1);

    std::cout << "[PASS] test_sampler_top_p\n";
}

void test_sampler_seed_reproducibility() {
    SamplingConfig cfg;
    cfg.temperature = 1.0f;
    cfg.top_k = 5;
    cfg.seed = 42;

    float logits[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 0.1f, 0.2f};
    int vocab_size = 7;

    std::vector<int> run1;
    Sampler s1(cfg);
    for (int i = 0; i < 50; ++i) {
        SampleResult res;
        Status st = s1.sample(logits, vocab_size, res);
        assert(st.is_ok());
        run1.push_back(res.token);
    }

    // Reset and run again with same seed
    std::vector<int> run2;
    Sampler s2(cfg);
    for (int i = 0; i < 50; ++i) {
        SampleResult res;
        Status st = s2.sample(logits, vocab_size, res);
        assert(st.is_ok());
        run2.push_back(res.token);
    }

    assert(run1 == run2);

    // Different seed should produce different sequence
    cfg.seed = 9999;
    std::vector<int> run3;
    Sampler s3(cfg);
    for (int i = 0; i < 50; ++i) {
        SampleResult res;
        Status st = s3.sample(logits, vocab_size, res);
        assert(st.is_ok());
        run3.push_back(res.token);
    }

    assert(run1 != run3);

    std::cout << "[PASS] test_sampler_seed_reproducibility\n";
}

void test_sampler_probability_sum_and_stability() {
    std::vector<std::pair<int, float>> candidates = {
        {0, -1000.0f},
        {1, -1002.0f},
        {2, -1005.0f}
    };

    std::vector<double> probs;
    Status st = Sampler::stable_softmax(candidates, probs);
    assert(st.is_ok());
    assert(probs.size() == 3);

    double sum = 0.0;
    for (double p : probs) {
        assert(p >= 0.0 && p <= 1.0);
        sum += p;
    }
    assert(std::fabs(sum - 1.0) < 1e-6);

    std::cout << "[PASS] test_sampler_probability_sum_and_stability\n";
}

int main() {
    std::cout << "Starting Sampler unit tests...\n";
    test_sampler_greedy();
    test_sampler_repetition_penalty();
    test_sampler_temperature();
    test_sampler_top_k();
    test_sampler_top_p();
    test_sampler_seed_reproducibility();
    test_sampler_probability_sum_and_stability();
    std::cout << "All Sampler tests passed!\n";
    return 0;
}

#ifndef INFERLET_ENGINE_RESULT_SINK_H_INCLUDED
#define INFERLET_ENGINE_RESULT_SINK_H_INCLUDED

#include "support/status.h"
#include "support/validation.h"
#include "core/task_output.h"

namespace inferlet {

struct TokenStep {
    int token = -1;
    std::string piece;
    int offset = 0;
    bool has_probability = false;
    float probability = 0.0f;
    std::vector<float> logits;
    std::vector<float> embedding;
};

class ResultSink {
public:
    virtual ~ResultSink() = default;
    virtual Status on_input(const InputTokenInfo &info) = 0;
    virtual Status on_token(TokenStep &&step) = 0;
    virtual void on_finish(Status status) = 0;
};

class CollectingSink : public ResultSink {
public:
    explicit CollectingSink(TaskOutputInternal &output, size_t max_verbose_bytes = 0)
        : output_(output), max_verbose_bytes_(max_verbose_bytes) {}

    Status on_input(const InputTokenInfo &info) override {
        output_.input = info;
        output_.input.update_ptrs();
        return Status::ok();
    }

    Status on_token(TokenStep &&step) override {
        size_t added_bytes = (step.logits.size() + step.embedding.size()) * sizeof(float);
        if (max_verbose_bytes_ > 0 && current_verbose_bytes_ + added_bytes > max_verbose_bytes_) {
            return Status::out_of_memory("Verbose memory budget exceeded during generation (limit: " +
                                         std::to_string(max_verbose_bytes_) + " bytes)");
        }
        current_verbose_bytes_ += added_bytes;

        output_.result.tokens.push_back(step.token);
        output_.result.byte_offsets.push_back(step.offset);
        if (step.has_probability) {
            output_.result.probabilities.push_back(step.probability);
        }
        output_.result.text.append(step.piece);

        if (!step.logits.empty()) {
            if (output_.result.logits.cols == 0) {
                output_.result.logits.cols = static_cast<int>(step.logits.size());
            }
            output_.result.logits.rows++;
            output_.result.logits.values.insert(output_.result.logits.values.end(),
                                                step.logits.begin(), step.logits.end());
        }
        if (!step.embedding.empty()) {
            if (output_.result.embeddings.cols == 0) {
                output_.result.embeddings.cols = static_cast<int>(step.embedding.size());
            }
            output_.result.embeddings.rows++;
            output_.result.embeddings.values.insert(output_.result.embeddings.values.end(),
                                                    step.embedding.begin(), step.embedding.end());
        }
        return Status::ok();
    }

    void on_finish(Status status) override {
        finish_status_ = status;
        // Token limits can end generation in the middle of a UTF-8 character.
        // Preserve earlier errors, and never publish malformed text as success.
        if (finish_status_.is_ok() && !is_valid_utf8(output_.result.text)) {
            finish_status_ = Status::inference_error("Incomplete or invalid UTF-8 sequence at end of generation");
        }
        if (output_.result.logits.rows > 0) {
            output_.result.logits.finalize_row_ptrs();
        }
        if (output_.result.embeddings.rows > 0) {
            output_.result.embeddings.finalize_row_ptrs();
        }
    }

    Status finish_status() const { return finish_status_; }

private:
    TaskOutputInternal &output_;
    size_t max_verbose_bytes_ = 0;
    size_t current_verbose_bytes_ = 0;
    Status finish_status_ = Status::ok();
};

} // namespace inferlet

#endif // INFERLET_ENGINE_RESULT_SINK_H_INCLUDED

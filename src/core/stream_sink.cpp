#include "stream_sink.h"
#include <cstdint>

namespace inferlet {

enum class Utf8CheckResult {
    Valid,       // Entire byte sequence is complete, valid UTF-8
    Incomplete,  // Valid UTF-8 prefix, but ends with incomplete multibyte sequence
    Invalid      // Contains invalid UTF-8 byte sequence
};

static Utf8CheckResult check_utf8_sequence(const std::string &bytes) {
    const uint8_t *data = reinterpret_cast<const uint8_t *>(bytes.data());
    size_t len = bytes.size();
    size_t i = 0;

    while (i < len) {
        uint8_t b0 = data[i];
        if (b0 <= 0x7F) {
            // 1-byte ASCII (00..7F)
            i += 1;
        } else if (b0 >= 0xC2 && b0 <= 0xDF) {
            // 2-byte sequence (C2..DF 80..BF)
            if (i + 1 >= len) {
                return Utf8CheckResult::Incomplete;
            }
            uint8_t b1 = data[i + 1];
            if (b1 < 0x80 || b1 > 0xBF) return Utf8CheckResult::Invalid;
            i += 2;
        } else if (b0 >= 0xE0 && b0 <= 0xEF) {
            // 3-byte sequence
            if (i + 1 >= len) {
                return Utf8CheckResult::Incomplete;
            }
            uint8_t b1 = data[i + 1];
            if (b0 == 0xE0) {
                if (b1 < 0xA0 || b1 > 0xBF) return Utf8CheckResult::Invalid;
            } else if (b0 == 0xED) {
                if (b1 < 0x80 || b1 > 0x9F) return Utf8CheckResult::Invalid;
            } else {
                if (b1 < 0x80 || b1 > 0xBF) return Utf8CheckResult::Invalid;
            }
            if (i + 2 >= len) {
                return Utf8CheckResult::Incomplete;
            }
            uint8_t b2 = data[i + 2];
            if (b2 < 0x80 || b2 > 0xBF) return Utf8CheckResult::Invalid;
            i += 3;
        } else if (b0 >= 0xF0 && b0 <= 0xF4) {
            // 4-byte sequence
            if (i + 1 >= len) {
                return Utf8CheckResult::Incomplete;
            }
            uint8_t b1 = data[i + 1];
            if (b0 == 0xF0) {
                if (b1 < 0x90 || b1 > 0xBF) return Utf8CheckResult::Invalid;
            } else if (b0 == 0xF4) {
                if (b1 < 0x80 || b1 > 0x8F) return Utf8CheckResult::Invalid;
            } else {
                if (b1 < 0x80 || b1 > 0xBF) return Utf8CheckResult::Invalid;
            }
            if (i + 2 >= len) {
                return Utf8CheckResult::Incomplete;
            }
            uint8_t b2 = data[i + 2];
            if (b2 < 0x80 || b2 > 0xBF) return Utf8CheckResult::Invalid;
            if (i + 3 >= len) {
                return Utf8CheckResult::Incomplete;
            }
            uint8_t b3 = data[i + 3];
            if (b3 < 0x80 || b3 > 0xBF) return Utf8CheckResult::Invalid;
            i += 4;
        } else {
            // Invalid leading byte (80..BF, C0..C1, F5..FF)
            return Utf8CheckResult::Invalid;
        }
    }
    return Utf8CheckResult::Valid;
}

StreamingSink::StreamingSink(std::shared_ptr<TaskCore> core, size_t max_verbose_bytes)
    : core_(std::move(core)), max_verbose_bytes_(max_verbose_bytes) {}

Status StreamingSink::on_input(const InputTokenInfo &info) {
    if (!info.tokens.empty()) {
        first_input_info_ = info;
        first_input_info_->update_ptrs();
    }
    return Status::ok();
}

Status StreamingSink::on_token(TokenStep &&step) {
    if (!core_ || core_->cancel_requested.load()) {
        return Status::cancelled("Task was cancelled");
    }

    size_t added_bytes = (step.logits.size() + step.embedding.size()) * sizeof(float);
    if (max_verbose_bytes_ > 0 && current_verbose_bytes_ + added_bytes > max_verbose_bytes_) {
        return Status::out_of_memory("Verbose memory budget exceeded during generation (limit: " +
                                     std::to_string(max_verbose_bytes_) + " bytes)");
    }
    current_verbose_bytes_ += added_bytes;

    int offset = static_cast<int>(pending_chunk_.result.text.size());
    pending_chunk_.result.tokens.push_back(step.token);
    pending_chunk_.result.byte_offsets.push_back(offset);
    if (step.has_probability) {
        pending_chunk_.result.probabilities.push_back(step.probability);
    }
    pending_chunk_.result.text.append(step.piece);

    if (!step.logits.empty()) {
        if (pending_chunk_.result.logits.cols == 0) {
            pending_chunk_.result.logits.cols = static_cast<int>(step.logits.size());
        }
        pending_chunk_.result.logits.rows++;
        pending_chunk_.result.logits.values.insert(pending_chunk_.result.logits.values.end(),
                                                  step.logits.begin(), step.logits.end());
    }
    if (!step.embedding.empty()) {
        if (pending_chunk_.result.embeddings.cols == 0) {
            pending_chunk_.result.embeddings.cols = static_cast<int>(step.embedding.size());
        }
        pending_chunk_.result.embeddings.rows++;
        pending_chunk_.result.embeddings.values.insert(pending_chunk_.result.embeddings.values.end(),
                                                      step.embedding.begin(), step.embedding.end());
    }

    if (pending_chunk_.result.text.empty()) {
        // Empty piece (e.g. special token), keep accumulating
        return Status::ok();
    }

    Utf8CheckResult res = check_utf8_sequence(pending_chunk_.result.text);
    if (res == Utf8CheckResult::Invalid) {
        return Status::inference_error("Invalid UTF-8 byte sequence encountered during generation");
    }
    if (res == Utf8CheckResult::Incomplete) {
        // Multi-byte character not complete yet, wait for next token piece
        return Status::ok();
    }

    return emit_pending_chunk();
}

Status StreamingSink::emit_pending_chunk() {
    if (pending_chunk_.result.tokens.empty() && pending_chunk_.result.text.empty()) {
        return Status::ok();
    }

    if (!core_) {
        return Status::invalid_state("StreamingSink has no TaskCore");
    }

    std::unique_lock<std::mutex> lock(core_->mutex);
    core_->queue_not_full.wait(lock, [this]() {
        return core_->cancel_requested.load() || core_->chunks.size() < core_->max_queued_chunks;
    });

    if (core_->cancel_requested.load()) {
        return Status::cancelled("Task was cancelled");
    }

    if (!first_chunk_emitted_) {
        if (first_input_info_.has_value()) {
            pending_chunk_.input = std::move(first_input_info_);
            pending_chunk_.input->update_ptrs();
            first_input_info_.reset();
        }
        first_chunk_emitted_ = true;
    }

    if (pending_chunk_.result.logits.rows > 0) {
        pending_chunk_.result.logits.finalize_row_ptrs();
    }
    if (pending_chunk_.result.embeddings.rows > 0) {
        pending_chunk_.result.embeddings.finalize_row_ptrs();
    }

    core_->chunks.push_back(std::move(pending_chunk_));
    pending_chunk_ = StreamChunk{};
    core_->cv.notify_one();
    return Status::ok();
}

void StreamingSink::on_finish(Status status) {
    if (!status.is_ok()) {
        finish_status_ = status;
        return;
    }

    if (!pending_chunk_.result.tokens.empty() || !pending_chunk_.result.text.empty()) {
        if (!pending_chunk_.result.text.empty()) {
            Utf8CheckResult res = check_utf8_sequence(pending_chunk_.result.text);
            if (res != Utf8CheckResult::Valid) {
                finish_status_ = Status::inference_error("Incomplete or invalid UTF-8 sequence at end of stream");
                return;
            }
        }
        Status st = emit_pending_chunk();
        if (!st.is_ok()) {
            finish_status_ = st;
        }
    }
}

} // namespace inferlet

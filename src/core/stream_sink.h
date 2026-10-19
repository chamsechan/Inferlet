#ifndef INFERLET_CORE_STREAM_SINK_H_INCLUDED
#define INFERLET_CORE_STREAM_SINK_H_INCLUDED

#include "engine/result_sink.h"
#include "core/inference_task.h"
#include <memory>
#include <string>
#include <vector>
#include <optional>

namespace inferlet {

class StreamingSink : public ResultSink {
public:
    explicit StreamingSink(std::shared_ptr<TaskCore> core, size_t max_verbose_bytes = 0);
    ~StreamingSink() override = default;

    Status on_input(const InputTokenInfo &info) override;
    Status on_token(TokenStep &&step) override;
    void on_finish(Status status) override;

    Status finish_status() const { return finish_status_; }

private:
    Status emit_pending_chunk();

    std::shared_ptr<TaskCore> core_;
    size_t max_verbose_bytes_ = 0;
    size_t current_verbose_bytes_ = 0;
    std::optional<InputTokenInfo> first_input_info_;
    StreamChunk pending_chunk_;
    bool first_chunk_emitted_ = false;
    Status finish_status_ = Status::ok();
};

} // namespace inferlet

#endif // INFERLET_CORE_STREAM_SINK_H_INCLUDED

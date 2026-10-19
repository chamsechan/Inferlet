#ifndef INFERLET_ENGINE_GENERATOR_H_INCLUDED
#define INFERLET_ENGINE_GENERATOR_H_INCLUDED

#include "support/status.h"
#include "core/task_input.h"
#include "core/model_registry.h"
#include "result_sink.h"
#include <atomic>

namespace inferlet {

class Generator {
public:
    static Status run(const TaskRequest &request,
                      const std::shared_ptr<SharedModel> &model,
                      ResultSink &sink,
                      const std::atomic<bool> *cancel_flag = nullptr);
};

} // namespace inferlet

#endif // INFERLET_ENGINE_GENERATOR_H_INCLUDED

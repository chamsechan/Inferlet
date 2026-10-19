#ifndef INFERLET_SUPPORT_STATUS_H_INCLUDED
#define INFERLET_SUPPORT_STATUS_H_INCLUDED

#include <string>
#include <string_view>
#include "inferlet.h"

namespace inferlet {

struct Status {
    int code = INFERLET_OK;
    std::string message;

    Status() = default;
    Status(int c, std::string msg = {}) : code(c), message(std::move(msg)) {}

    static Status ok() {
        return Status(INFERLET_OK);
    }
    static Status invalid_argument(std::string msg = "Invalid argument") {
        return Status(INFERLET_E_INVALID_ARGUMENT, std::move(msg));
    }
    static Status invalid_handle(std::string msg = "Invalid handle") {
        return Status(INFERLET_E_INVALID_HANDLE, std::move(msg));
    }
    static Status invalid_state(std::string msg = "Invalid state") {
        return Status(INFERLET_E_INVALID_STATE, std::move(msg));
    }
    static Status io_error(std::string msg = "I/O error") {
        return Status(INFERLET_E_IO, std::move(msg));
    }
    static Status not_found(std::string msg = "Not found") {
        return Status(INFERLET_E_IO, std::move(msg));
    }
    static Status model_load(std::string msg = "Model load failed") {
        return Status(INFERLET_E_MODEL_LOAD, std::move(msg));
    }
    static Status inference_error(std::string msg = "Inference failed") {
        return Status(INFERLET_E_INFERENCE, std::move(msg));
    }
    static Status unsupported(std::string msg = "Unsupported feature or configuration") {
        return Status(INFERLET_E_UNSUPPORTED, std::move(msg));
    }
    static Status cancelled(std::string msg = "Operation cancelled") {
        return Status(INFERLET_E_CANCELLED, std::move(msg));
    }
    static Status buffer_too_small(std::string msg = "Buffer too small") {
        return Status(INFERLET_E_BUFFER_TOO_SMALL, std::move(msg));
    }
    static Status out_of_memory(std::string msg = "Out of memory") {
        return Status(INFERLET_E_OUT_OF_MEMORY, std::move(msg));
    }
    static Status internal(std::string msg = "Internal error") {
        return Status(INFERLET_E_INTERNAL, std::move(msg));
    }

    bool is_ok() const {
        return code == INFERLET_OK;
    }
    bool is_error() const {
        return !is_ok();
    }
    explicit operator bool() const {
        return is_ok();
    }
};

} // namespace inferlet

#endif // INFERLET_SUPPORT_STATUS_H_INCLUDED

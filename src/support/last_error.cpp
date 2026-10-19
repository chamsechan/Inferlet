#include "last_error.h"
#include <string>

namespace inferlet {

struct ThreadErrorState {
    int code = 0;
    std::string message;
};

static thread_local ThreadErrorState tls_last_error;

void set_last_error(int code, std::string_view message) {
    tls_last_error.code = code;
    tls_last_error.message = std::string(message);
}

void clear_last_error() {
    tls_last_error.code = 0;
    tls_last_error.message.clear();
}

const char *get_last_error() {
    return tls_last_error.message.c_str();
}

int get_last_error_code() {
    return tls_last_error.code;
}

int publish_status(const Status &st) {
    if (st.is_ok()) {
        clear_last_error();
        return 0;
    }
    set_last_error(st.code, st.message);
    return st.code;
}

int publish_error(int code, std::string_view message) {
    set_last_error(code, message);
    return code;
}

int publish_error(const Status &st) {
    return publish_status(st);
}

} // namespace inferlet

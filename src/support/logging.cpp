#include "logging.h"
#include <iostream>
#include <mutex>

namespace inferlet {

static LogLevel g_log_level = LogLevel::Info;
static std::mutex g_log_mutex;

void set_log_level(LogLevel level) {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    g_log_level = level;
}

LogLevel get_log_level() {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    return g_log_level;
}

static void log_message(LogLevel level, std::string_view prefix, std::string_view msg) {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    if (level < g_log_level) {
        return;
    }
    std::cerr << "[Inferlet " << prefix << "] " << msg << "\n";
}

void log_debug(std::string_view msg) {
    log_message(LogLevel::Debug, "DEBUG", msg);
}

void log_info(std::string_view msg) {
    log_message(LogLevel::Info, "INFO", msg);
}

void log_warn(std::string_view msg) {
    log_message(LogLevel::Warn, "WARN", msg);
}

void log_error(std::string_view msg) {
    log_message(LogLevel::Error, "ERROR", msg);
}

} // namespace inferlet

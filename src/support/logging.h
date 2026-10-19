#ifndef INFERLET_SUPPORT_LOGGING_H_INCLUDED
#define INFERLET_SUPPORT_LOGGING_H_INCLUDED

#include <string_view>

namespace inferlet {

enum class LogLevel {
    Debug = 0,
    Info = 1,
    Warn = 2,
    Error = 3,
    Off = 4
};

void set_log_level(LogLevel level);
LogLevel get_log_level();

void log_debug(std::string_view msg);
void log_info(std::string_view msg);
void log_warn(std::string_view msg);
void log_error(std::string_view msg);

} // namespace inferlet

#endif // INFERLET_SUPPORT_LOGGING_H_INCLUDED

#ifndef INFERLET_SUPPORT_LAST_ERROR_H_INCLUDED
#define INFERLET_SUPPORT_LAST_ERROR_H_INCLUDED

#include "status.h"
#include <string_view>

namespace inferlet {

void set_last_error(int code, std::string_view message);
void clear_last_error();
const char *get_last_error();
int get_last_error_code();

int publish_status(const Status &st);
int publish_error(int code, std::string_view message);
int publish_error(const Status &st);

} // namespace inferlet

#endif // INFERLET_SUPPORT_LAST_ERROR_H_INCLUDED

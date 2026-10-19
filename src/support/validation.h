#ifndef INFERLET_SUPPORT_VALIDATION_H_INCLUDED
#define INFERLET_SUPPORT_VALIDATION_H_INCLUDED

#include <cstddef>
#include <string_view>
#include <limits>

namespace inferlet {

bool is_valid_utf8(std::string_view str);
bool is_valid_json(std::string_view str);
bool is_valid_json_object(std::string_view str);

template <typename T>
inline bool safe_multiply(T a, T b, T &out) {
    if (a == 0 || b == 0) {
        out = 0;
        return true;
    }
    if (a > (std::numeric_limits<T>::max() / b)) {
        return false;
    }
    out = a * b;
    return true;
}

} // namespace inferlet

#endif // INFERLET_SUPPORT_VALIDATION_H_INCLUDED

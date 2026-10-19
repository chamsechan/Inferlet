#include "validation.h"
#include <nlohmann/json.hpp>

namespace inferlet {

bool is_valid_utf8(std::string_view str) {
    const auto *bytes = reinterpret_cast<const unsigned char *>(str.data());
    size_t len = str.size();
    size_t i = 0;

    while (i < len) {
        unsigned char c = bytes[i];
        if (c <= 0x7F) {
            i += 1;
        } else if ((c & 0xE0) == 0xC0) {
            if (i + 1 >= len) return false;
            if ((bytes[i + 1] & 0xC0) != 0x80) return false;
            if (c < 0xC2) return false; // Overlong
            i += 2;
        } else if ((c & 0xF0) == 0xE0) {
            if (i + 2 >= len) return false;
            if ((bytes[i + 1] & 0xC0) != 0x80 || (bytes[i + 2] & 0xC0) != 0x80) return false;
            if (c == 0xE0 && bytes[i + 1] < 0xA0) return false; // Overlong
            if (c == 0xED && bytes[i + 1] >= 0xA0) return false; // Surrogate
            i += 3;
        } else if ((c & 0xF8) == 0xF0) {
            if (i + 3 >= len) return false;
            if ((bytes[i + 1] & 0xC0) != 0x80 ||
                (bytes[i + 2] & 0xC0) != 0x80 ||
                (bytes[i + 3] & 0xC0) != 0x80) return false;
            if (c == 0xF0 && bytes[i + 1] < 0x90) return false; // Overlong
            if (c == 0xF4 && bytes[i + 1] > 0x8F) return false; // Out of Unicode range
            if (c > 0xF4) return false;
            i += 4;
        } else {
            return false;
        }
    }
    return true;
}

bool is_valid_json(std::string_view str) {
    nlohmann::json j = nlohmann::json::parse(str, nullptr, false);
    return !j.is_discarded();
}

bool is_valid_json_object(std::string_view str) {
    nlohmann::json j = nlohmann::json::parse(str, nullptr, false);
    return !j.is_discarded() && j.is_object();
}

} // namespace inferlet

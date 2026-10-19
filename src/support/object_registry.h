#ifndef INFERLET_SUPPORT_OBJECT_REGISTRY_H_INCLUDED
#define INFERLET_SUPPORT_OBJECT_REGISTRY_H_INCLUDED

#include <cstdint>
#include <mutex>
#include <unordered_map>
#include "inferlet.h"
#include "status.h"

namespace inferlet {

enum class ObjectKind : uint32_t {
    Parameter = 1,
    ModelHandle = 2,
    TaskInput = 3,
    TaskTicket = 4,
    StreamTicket = 5,
    TaskOutput = 6
};

constexpr uint64_t INFERLET_OBJECT_MAGIC = 0x494E4645524C4554ULL; // 'INFERLET'
constexpr uint32_t INFERLET_ABI_VERSION = 1;

struct ObjectHeader {
    uint64_t magic = INFERLET_OBJECT_MAGIC;
    ObjectKind kind = ObjectKind::Parameter;
    uint32_t abi_version = INFERLET_ABI_VERSION;
};

class LiveObjectRegistry {
public:
    static LiveObjectRegistry &instance();

    void register_object(const void *ptr, ObjectKind kind);
    void unregister_object(const void *ptr);
    bool is_valid(const void *ptr, ObjectKind expected_kind) const;
    bool is_registered(const void *ptr) const;
    void clear();

private:
    LiveObjectRegistry() = default;
    ~LiveObjectRegistry() = default;
    LiveObjectRegistry(const LiveObjectRegistry &) = delete;
    LiveObjectRegistry &operator=(const LiveObjectRegistry &) = delete;

    mutable std::mutex mutex_;
    std::unordered_map<const void *, ObjectKind> objects_;
};

template <typename T>
T *checked_cast(void *ptr, ObjectKind expected_kind) {
    if (!ptr) {
        return nullptr;
    }
    if (!LiveObjectRegistry::instance().is_valid(ptr, expected_kind)) {
        return nullptr;
    }
    auto *header = reinterpret_cast<ObjectHeader *>(ptr);
    if (header->magic != INFERLET_OBJECT_MAGIC || header->kind != expected_kind) {
        return nullptr;
    }
    return reinterpret_cast<T *>(ptr);
}

} // namespace inferlet

#endif // INFERLET_SUPPORT_OBJECT_REGISTRY_H_INCLUDED

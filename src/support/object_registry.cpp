#include "object_registry.h"

namespace inferlet {

LiveObjectRegistry &LiveObjectRegistry::instance() {
    static LiveObjectRegistry instance;
    return instance;
}

void LiveObjectRegistry::register_object(const void *ptr, ObjectKind kind) {
    if (!ptr) return;
    std::lock_guard<std::mutex> lock(mutex_);
    objects_[ptr] = kind;
}

void LiveObjectRegistry::unregister_object(const void *ptr) {
    if (!ptr) return;
    std::lock_guard<std::mutex> lock(mutex_);
    objects_.erase(ptr);
}

bool LiveObjectRegistry::is_valid(const void *ptr, ObjectKind expected_kind) const {
    if (!ptr) return false;
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = objects_.find(ptr);
    if (it == objects_.end()) {
        return false;
    }
    return it->second == expected_kind;
}

bool LiveObjectRegistry::is_registered(const void *ptr) const {
    if (!ptr) return false;
    std::lock_guard<std::mutex> lock(mutex_);
    return objects_.find(ptr) != objects_.end();
}

void LiveObjectRegistry::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    objects_.clear();
}

} // namespace inferlet

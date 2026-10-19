#include "visual_cache.h"
#include <algorithm>

namespace inferlet {

VisualCache::VisualCache(size_t max_bytes) : max_bytes_(max_bytes > 0 ? max_bytes : 1073741824) {
}

uint64_t VisualCache::compute_hash(const void *data, size_t size) {
    if (!data || size == 0) return 0;
    const uint8_t *ptr = static_cast<const uint8_t *>(data);
    uint64_t h = 14695981039346656037ULL;
    for (size_t i = 0; i < size; ++i) {
        h ^= ptr[i];
        h *= 1099511628211ULL;
    }
    return h;
}

Status VisualCache::lookup(const std::string &key,
                           uint64_t data_hash,
                           int dtype,
                           int grid_t,
                           int grid_h,
                           int grid_w,
                           std::shared_ptr<const std::vector<float>> &out_embeddings) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = cache_map_.find(key);
    if (it == cache_map_.end()) {
        return Status::not_found("Visual cache key not found: " + key);
    }

    auto &item = it->second.entry;

    // Check metadata and data hash consistency
    if (item.data_hash != data_hash || item.dtype != dtype ||
        item.grid_t != grid_t || item.grid_h != grid_h || item.grid_w != grid_w) {
        return Status::invalid_argument("Inconsistent visual cache data for key: " + key);
    }

    // Move to front of LRU
    lru_order_.erase(it->second.lru_iter);
    lru_order_.push_front(key);
    it->second.lru_iter = lru_order_.begin();

    out_embeddings = item.embeddings;

    // Decrement remaining hits
    item.remaining_hits--;
    if (item.remaining_hits <= 0) {
        // Budget exhausted; remove from cache map
        current_bytes_ -= item.byte_size;
        lru_order_.erase(it->second.lru_iter);
        cache_map_.erase(it);
    }

    return Status::ok();
}

Status VisualCache::insert(const std::string &key,
                           uint64_t data_hash,
                           int dtype,
                           int grid_t,
                           int grid_h,
                           int grid_w,
                           std::shared_ptr<const std::vector<float>> embeddings,
                           int cache_times) {
    if (key.empty()) {
        return Status::invalid_argument("Visual cache key cannot be empty");
    }
    if (cache_times < 2) {
        return Status::invalid_argument("cache_times must be >= 2 for cached visual tensors");
    }
    if (!embeddings) {
        return Status::invalid_argument("Visual cache embeddings cannot be NULL");
    }

    size_t byte_size = embeddings->size() * sizeof(float);
    if (byte_size > max_bytes_) {
        return Status::invalid_argument("Visual cache item size (" + std::to_string(byte_size) +
                                       " bytes) exceeds cache limit (" + std::to_string(max_bytes_) + " bytes)");
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // If key already exists, check consistency
    auto it = cache_map_.find(key);
    if (it != cache_map_.end()) {
        if (it->second.entry.data_hash != data_hash || it->second.entry.dtype != dtype ||
            it->second.entry.grid_t != grid_t || it->second.entry.grid_h != grid_h || it->second.entry.grid_w != grid_w) {
            return Status::invalid_argument("Inconsistent visual cache data for existing key: " + key);
        }
        // Update remaining hits
        it->second.entry.remaining_hits = std::max(it->second.entry.remaining_hits, cache_times - 1);
        // Move to front of LRU
        lru_order_.erase(it->second.lru_iter);
        lru_order_.push_front(key);
        it->second.lru_iter = lru_order_.begin();
        return Status::ok();
    }

    // Evict entries if needed
    evict_lru_under_lock(byte_size);

    VisualCacheEntry entry;
    entry.key = key;
    entry.data_hash = data_hash;
    entry.dtype = dtype;
    entry.grid_t = grid_t;
    entry.grid_h = grid_h;
    entry.grid_w = grid_w;
    entry.remaining_hits = cache_times - 1;
    entry.byte_size = byte_size;
    entry.embeddings = embeddings;

    lru_order_.push_front(key);
    MapValue mv;
    mv.entry = std::move(entry);
    mv.lru_iter = lru_order_.begin();

    cache_map_.emplace(key, std::move(mv));
    current_bytes_ += byte_size;

    return Status::ok();
}

void VisualCache::evict_lru_under_lock(size_t required_bytes) {
    while (current_bytes_ + required_bytes > max_bytes_ && !lru_order_.empty()) {
        std::string oldest_key = lru_order_.back();
        auto it = cache_map_.find(oldest_key);
        if (it != cache_map_.end()) {
            current_bytes_ -= it->second.entry.byte_size;
            cache_map_.erase(it);
        }
        lru_order_.pop_back();
    }
}

size_t VisualCache::current_bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return current_bytes_;
}

size_t VisualCache::entry_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cache_map_.size();
}

void VisualCache::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_map_.clear();
    lru_order_.clear();
    current_bytes_ = 0;
}

} // namespace inferlet

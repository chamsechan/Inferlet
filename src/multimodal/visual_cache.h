#ifndef INFERLET_MULTIMODAL_VISUAL_CACHE_H_INCLUDED
#define INFERLET_MULTIMODAL_VISUAL_CACHE_H_INCLUDED

#include "support/status.h"
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <list>
#include <cstdint>

namespace inferlet {

struct VisualCacheEntry {
    std::string key;
    uint64_t data_hash = 0;
    int dtype = 0;
    int grid_t = 0;
    int grid_h = 0;
    int grid_w = 0;
    int remaining_hits = 0;
    size_t byte_size = 0;
    std::shared_ptr<const std::vector<float>> embeddings;
};

class VisualCache {
public:
    explicit VisualCache(size_t max_bytes = 1073741824); // 1GB default
    ~VisualCache() = default;

    Status lookup(const std::string &key,
                  uint64_t data_hash,
                  int dtype,
                  int grid_t,
                  int grid_h,
                  int grid_w,
                  std::shared_ptr<const std::vector<float>> &out_embeddings);

    Status insert(const std::string &key,
                  uint64_t data_hash,
                  int dtype,
                  int grid_t,
                  int grid_h,
                  int grid_w,
                  std::shared_ptr<const std::vector<float>> embeddings,
                  int cache_times);

    size_t current_bytes() const;
    size_t entry_count() const;
    void clear();

    static uint64_t compute_hash(const void *data, size_t size);

private:
    size_t max_bytes_;
    mutable std::mutex mutex_;
    size_t current_bytes_ = 0;

    // LRU tracking: list of keys, front is most recently used
    std::list<std::string> lru_order_;
    using ListIter = std::list<std::string>::iterator;

    struct MapValue {
        VisualCacheEntry entry;
        ListIter lru_iter;
    };

    std::unordered_map<std::string, MapValue> cache_map_;

    void evict_lru_under_lock(size_t required_bytes);
};

} // namespace inferlet

#endif // INFERLET_MULTIMODAL_VISUAL_CACHE_H_INCLUDED

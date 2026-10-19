#include "inferlet.h"
#include "multimodal/multimodal_backend.h"
#include "multimodal/visual_cache.h"

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

using namespace inferlet;

static const char *get_model_path() {
    const char *path = std::getenv("INFERLET_TEST_MODEL");
    return path && std::strlen(path) > 0
               ? path
               : "/home/ubuntu/project/llm-ops-agy/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
}

static void test_shape_validation() {
    VisualEncoderMetadata metadata;
    metadata.channels = 3;
    metadata.patch_size = 14;
    metadata.temporal_patch_size = 2;
    metadata.n_embd = 896;

    size_t product = 0;
    assert(MultimodalBackend::checked_mul(100, 200, product) && product == 20000);
    assert(!MultimodalBackend::checked_mul(std::numeric_limits<size_t>::max() / 2 + 1, 3, product));

    MediaTensor tensor;
    tensor.type = MediaType::Image;
    tensor.dtype = DataType::UInt8;
    tensor.grid_t = 1;
    tensor.grid_h = 1;
    tensor.grid_w = 1;
    tensor.owned_data.resize(3 * 14 * 14, 128);
    assert(MultimodalBackend::validate_tensor(tensor, metadata).is_ok());
    tensor.owned_data.push_back(0);
    assert(MultimodalBackend::validate_tensor(tensor, metadata).is_error());
}

static void test_visual_cache() {
    VisualCache cache(10 * 1024);
    auto first = std::make_shared<std::vector<float>>(1500, 1.0f);
    auto second = std::make_shared<std::vector<float>>(1500, 2.0f);
    assert(cache.insert("first", 1, 1, 1, 1, 1, first, 3).is_ok());
    assert(cache.insert("second", 2, 1, 1, 1, 1, second, 3).is_ok());
    assert(cache.entry_count() == 1);

    std::shared_ptr<const std::vector<float>> output;
    assert(!cache.lookup("first", 1, 1, 1, 1, 1, output).is_ok());
    assert(cache.lookup("second", 2, 1, 1, 1, 1, output).is_ok());
}

static void test_public_multimodal_rejection() {
    inferlet_init();
    void *model = inferlet_load_from_file(get_model_path(), nullptr);
    assert(model != nullptr);

    void *legacy_input = inferlet_task_input_allocate();
    std::vector<float> feature(896, 0.05f);
    inferlet_multimodal legacy{};
    legacy.prompt_text = "Describe this object";
    legacy.feature = feature.data();
    legacy.size = static_cast<int>(feature.size() * sizeof(float));
    assert(inferlet_task_input_set_multimodal(legacy_input, &legacy, 1) == INFERLET_OK);
    void *output = nullptr;
    assert(inferlet_run(model, legacy_input, &output) == INFERLET_E_UNSUPPORTED);
    assert(output == nullptr);
    inferlet_task_input_deallocate(legacy_input);

    void *chat_input = inferlet_task_input_allocate();
    std::vector<unsigned char> image(3 * 14 * 14, 128);
    inferlet_multimodal_chat_content contents[2]{};
    contents[0].type = "text";
    contents[0].data = "What is in this image?";
    contents[1].type = "image";
    contents[1].data = image.data();
    contents[1].size = static_cast<int>(image.size());
    contents[1].data_type = 4;
    contents[1].grid_t = 1;
    contents[1].grid_h = 1;
    contents[1].grid_w = 1;
    inferlet_multimodal_chat_history history{};
    history.role = "user";
    history.content = contents;
    history.length = 2;
    assert(inferlet_task_input_set_multimodal_chat_history(chat_input, &history, 1) == INFERLET_OK);
    assert(inferlet_run(model, chat_input, &output) == INFERLET_E_UNSUPPORTED);
    assert(output == nullptr);
    inferlet_task_input_deallocate(chat_input);

    inferlet_unload(model);
    inferlet_deinit();
}

int main() {
    test_shape_validation();
    test_visual_cache();
    test_public_multimodal_rejection();
    std::cout << "All multimodal boundary tests passed successfully!\n";
    return 0;
}

#ifndef INFERLET_CORE_TASK_OUTPUT_H_INCLUDED
#define INFERLET_CORE_TASK_OUTPUT_H_INCLUDED

#include "support/object_registry.h"
#include "engine/prompt_builder.h"
#include <vector>
#include <string>

namespace inferlet {

struct MatrixF32 {
    int rows = 0;
    int cols = 0;
    std::vector<float> values;
    std::vector<const float *> row_ptrs;

    void finalize_row_ptrs() {
        if (rows <= 0 || cols <= 0 || values.empty()) {
            row_ptrs.clear();
            return;
        }
        row_ptrs.resize(rows);
        for (int r = 0; r < rows; ++r) {
            row_ptrs[r] = values.data() + (r * cols);
        }
    }
};

struct TokenResult {
    std::string text;
    std::vector<int> tokens;
    std::vector<int> byte_offsets;
    std::vector<float> probabilities;
    MatrixF32 logits;
    MatrixF32 embeddings;
};

struct TaskOutputInternal {
    ObjectHeader header{INFERLET_OBJECT_MAGIC, ObjectKind::TaskOutput, INFERLET_ABI_VERSION};
    TokenResult result;
    InputTokenInfo input;
};

} // namespace inferlet

#endif // INFERLET_CORE_TASK_OUTPUT_H_INCLUDED

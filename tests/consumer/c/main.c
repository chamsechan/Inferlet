#include "inferlet.h"
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>

static void test_model(const char *path) {
    void *param = inferlet_parameter_allocate();
    inferlet_parameter_set_load_from_file_sync(param, 1);
    void *handle = inferlet_load_from_file(path, param);
    inferlet_parameter_deallocate(param);
    assert(handle != NULL);

    int tokens[64], count = 0, bytes = 0;
    char decoded[64];
    assert(inferlet_tokenizer_encode(handle, "Hello", 5, tokens, 64, &count, 0, 0) == INFERLET_OK);
    assert(count > 0);
    assert(inferlet_tokenizer_decode(handle, tokens, count, decoded, 64, &bytes, 1) == INFERLET_OK);
    assert(bytes == 5 && strcmp(decoded, "Hello") == 0);

    void *input = inferlet_task_input_allocate();
    assert(inferlet_task_input_set_prompt_tokens(input, tokens, count) == INFERLET_OK);
    assert(inferlet_task_input_set_max_output_tokens(input, 4) == INFERLET_OK);
    assert(inferlet_task_input_set_temperature(input, 0.0f) == INFERLET_OK);
    void *output = NULL;
    assert(inferlet_run(handle, input, &output) == INFERLET_OK && output != NULL);
    assert(inferlet_task_output_get_result_text(output) != NULL);
    assert(inferlet_task_output_get_result_tokens(output, &count) != NULL && count > 0);
    inferlet_task_output_deallocate(output);

    void *task = NULL;
    assert(inferlet_run_async(handle, input, &task) == INFERLET_OK && task != NULL);
    assert(inferlet_wait_task(handle, task, &output) == INFERLET_OK && output != NULL);
    inferlet_task_output_deallocate(output);

    assert(inferlet_task_input_set_stream_block_mode(input, 1) == INFERLET_OK);
    assert(inferlet_run_stream(handle, input, &task) == INFERLET_OK && task != NULL);
    inferlet_task_input_deallocate(input);
    int total = 0;
    const char *chunk;
    while ((chunk = inferlet_get_stream_text(handle, task)) != NULL) {
        if (*chunk) {
            assert(inferlet_get_stream_tokens(task, &count) != NULL && count > 0);
            total += count;
        }
    }
    assert(total > 0 && total <= 4);
    inferlet_unload(handle);
    printf("[consumer_c] Tokenizer, sync, async and stream inference passed.\n");
}

int main(void) {
    printf("[consumer_c] Testing Inferlet C ABI consumer...\n");

    inferlet_init();

    void *param = inferlet_parameter_allocate();
    assert(param != NULL);

    inferlet_parameter_set_decoder_max_token_cnt(param, 512);
    inferlet_parameter_set_device_id(param, 0);
    inferlet_parameter_set_tokenizer_slow(param, 0);
    inferlet_parameter_deallocate(param);

    void *task_in = inferlet_task_input_allocate();
    assert(task_in != NULL);

    inferlet_task_input_set_prompt_text(task_in, "C consumer smoke test");
    inferlet_task_input_set_max_output_tokens(task_in, 16);
    inferlet_task_input_deallocate(task_in);

    const char *model_path = getenv("INFERLET_TEST_MODEL");
    if (model_path && *model_path) test_model(model_path);

    inferlet_deinit();

    printf("[consumer_c] C consumer test PASSED!\n");
    return 0;
}

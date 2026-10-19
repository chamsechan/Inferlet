/* Uses only the installed C header and libinferlet.a. Test fixture: SmolVLM-256M. */
#include "inferlet.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void *image_input(int channel, const char *key, int bad_shape) {
    const int side = 256, plane = side * side;
    unsigned char *pixels = (unsigned char *)calloc(3 * plane, 1);
    assert(pixels);
    memset(pixels + channel * plane, 255, plane);
    inferlet_multimodal_chat_content content[2] = {0};
    content[0].type = "image";
    content[0].data = pixels;
    content[0].size = 3 * plane - bad_shape;
    content[0].grid_t = 1;
    content[0].grid_h = content[0].grid_w = side / 16;
    content[0].data_type = 4;
    content[0].cache_key = key;
    content[0].cache_times = key ? 8 : 0;
    content[1].type = "text";
    content[1].data = "What color is this image? Answer with the color only.";
    inferlet_multimodal_chat_history history = {0};
    history.role = "user";
    history.content = content;
    history.length = 2;
    void *input = inferlet_task_input_allocate();
    assert(input);
    assert(inferlet_task_input_set_multimodal_chat_history(input, &history, 1) == INFERLET_OK);
    /* The setter must own its copy, including asynchronous use. */
    memset(pixels, 0, 3 * plane);
    free(pixels);
    assert(inferlet_task_input_set_temperature(input, 0) == INFERLET_OK);
    assert(inferlet_task_input_set_max_output_tokens(input, 16) == INFERLET_OK);
    assert(inferlet_task_input_set_verbose_data_flag(input, INFERLET_VERBOSE_F_INPUT_TOKENS | INFERLET_VERBOSE_F_OUTPUT_LOGITS) == INFERLET_OK);
    return input;
}

static void *history_input(void) {
    const int plane = 256 * 256;
    unsigned char *red = (unsigned char *)calloc(3 * plane, 1);
    unsigned char *blue = (unsigned char *)calloc(3 * plane, 1);
    assert(red && blue);
    memset(red, 255, plane);
    memset(blue + 2 * plane, 255, plane);
    inferlet_multimodal_chat_content first[2] = {0}, last[2] = {0}, reply = {0};
    first[0].type = "image"; first[0].data = red; first[0].size = 3 * plane;
    first[0].data_type = 4; first[0].grid_t = 1; first[0].grid_h = first[0].grid_w = 16;
    first[0].cache_key = "red"; first[0].cache_times = 8;
    first[1].type = "text"; first[1].data = "What color is this image? Answer with the color only.";
    memcpy(last, first, sizeof(first));
    last[0].data = blue; last[0].cache_key = "blue";
    reply.type = "text"; reply.data = "Red.";
    inferlet_multimodal_chat_history history[3] = {0};
    history[0].role = "user"; history[0].content = first; history[0].length = 2;
    history[1].role = "assistant"; history[1].content = &reply; history[1].length = 1;
    history[2].role = "user"; history[2].content = last; history[2].length = 2;
    void *input = inferlet_task_input_allocate();
    assert(inferlet_task_input_set_multimodal_chat_history(input, history, 3) == INFERLET_OK);
    free(red); free(blue);
    assert(inferlet_task_input_set_temperature(input, 0) == INFERLET_OK);
    assert(inferlet_task_input_set_max_output_tokens(input, 16) == INFERLET_OK);
    assert(inferlet_task_input_set_verbose_data_flag(input, INFERLET_VERBOSE_F_INPUT_TOKENS) == INFERLET_OK);
    return input;
}

static void check_rejection(void *model, void *input, int expected) {
    void *output = NULL;
    int rc = inferlet_run(model, input, &output);
    if (rc != expected) fprintf(stderr, "Expected error %d, got %d\n", expected, rc);
    assert(rc == expected && !output);
    inferlet_task_input_deallocate(input);
}

int main(void) {
    const char *model_path = getenv("INFERLET_TEST_VISION_MODEL");
    const char *config_path = getenv("INFERLET_TEST_VISION_CONFIG");
    if (!model_path || !config_path) {
        puts("SKIP: set INFERLET_TEST_VISION_MODEL and INFERLET_TEST_VISION_CONFIG (SmolVLM-256M)");
        return 77;
    }
    inferlet_init();
    void *param = inferlet_parameter_allocate();
    inferlet_parameter_set_run_config_file(param, config_path);
    inferlet_parameter_set_load_from_file_sync(param, 1);
    void *model = inferlet_load_from_file(model_path, param);
    assert(model);
    inferlet_parameter_deallocate(param);

    void *red = image_input(0, "red", 0);
    void *blue = image_input(2, "blue", 0);
    void *red_out = NULL, *blue_out = NULL;
    int rc = inferlet_run(model, red, &red_out);
    fprintf(stderr, "Red image run status: %d\n", rc);
    assert(rc == INFERLET_OK && red_out);
    assert(inferlet_run(model, blue, &blue_out) == INFERLET_OK && blue_out);
    const char *red_text = inferlet_task_output_get_result_text(red_out);
    const char *blue_text = inferlet_task_output_get_result_text(blue_out);
    assert(red_text && *red_text && blue_text && *blue_text);
    printf("RED: %s\nBLUE: %s\n", red_text, blue_text);
    assert(strcmp(red_text, blue_text) != 0);
    assert(strstr(red_text, "red") || strstr(red_text, "Red"));
    assert(strstr(blue_text, "blue") || strstr(blue_text, "Blue"));
    int length = 0, vocab = 0;
    const char **pieces = NULL;
    const int *tokens = inferlet_task_output_get_input_tokens(red_out, &length, &pieces);
    assert(tokens && pieces && length > 0);
    int image_slots = 0;
    for (int i = 0; i < length; ++i) {
        if (tokens[i] == -1) { ++image_slots; assert(strcmp(pieces[i], "<image>") == 0); }
        else assert(tokens[i] >= 0);
    }
    assert(image_slots > 0);
    const float **logits = inferlet_task_output_get_output_logits(red_out, &length, &vocab);
    assert(logits && length > 0 && vocab > 0);

    void *task = NULL, *async_out = NULL;
    assert(inferlet_run_async(model, red, &task) == INFERLET_OK);
    inferlet_task_input_deallocate(red);
    assert(inferlet_wait_task(model, task, &async_out) == INFERLET_OK && async_out);
    assert(strcmp(inferlet_task_output_get_result_text(async_out), red_text) == 0);
    inferlet_task_output_deallocate(async_out);

    red = image_input(0, "red", 0);
    assert(inferlet_task_input_set_stream_block_mode(red, 1) == INFERLET_OK);
    assert(inferlet_run_stream(model, red, &task) == INFERLET_OK);
    inferlet_task_input_deallocate(red);
    char streamed[1024] = {0};
    const char *chunk;
    while ((chunk = inferlet_get_stream_text(model, task)) != NULL) {
        assert(strlen(streamed) + strlen(chunk) < sizeof(streamed));
        strcat(streamed, chunk);
    }
    assert(strcmp(streamed, red_text) == 0);

    void *history = history_input(), *history_out = NULL;
    assert(inferlet_run(model, history, &history_out) == INFERLET_OK && history_out);
    const char *history_text = inferlet_task_output_get_result_text(history_out);
    assert(history_text && (strstr(history_text, "Blue") || strstr(history_text, "blue")));
    printf("MULTI-IMAGE HISTORY: %s\n", history_text);
    tokens = inferlet_task_output_get_input_tokens(history_out, &length, &pieces);
    int history_image_slots = 0;
    for (int i = 0; i < length; ++i) if (tokens[i] == -1) ++history_image_slots;
    assert(history_image_slots == 2 * image_slots);
    inferlet_task_input_deallocate(history);
    inferlet_task_output_deallocate(history_out);

    /* Same cache key with different bytes must fail, not reuse another image. */
    check_rejection(model, image_input(2, "red", 0), INFERLET_E_INVALID_ARGUMENT);
    check_rejection(model, image_input(0, NULL, 1), INFERLET_E_INVALID_ARGUMENT);
    red = image_input(0, NULL, 0);
    assert(inferlet_task_input_set_beam_size(red, 2) == INFERLET_OK);
    check_rejection(model, red, INFERLET_E_UNSUPPORTED);
    red = image_input(0, NULL, 0);
    assert(inferlet_task_input_set_visual_encoder(red, "missing") == INFERLET_OK);
    check_rejection(model, red, INFERLET_E_UNSUPPORTED);
    red = image_input(0, NULL, 0);
    assert(inferlet_run_async(model, red, &task) == INFERLET_OK);
    assert(inferlet_cancel_task(model, task) == INFERLET_OK);
    async_out = NULL;
    assert(inferlet_wait_task(model, task, &async_out) == INFERLET_E_CANCELLED && !async_out);
    inferlet_task_input_deallocate(red);
    inferlet_task_input_deallocate(blue);
    inferlet_task_output_deallocate(red_out);
    inferlet_task_output_deallocate(blue_out);
    inferlet_unload(model);
    inferlet_deinit();
    puts("Vision C integration passed: image content, copied buffers, cache, sync/async/stream, cancellation and errors.");
    return 0;
}

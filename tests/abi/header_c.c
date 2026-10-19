#include "inferlet.h"
#include <stddef.h>
#include <stdio.h>
#include <assert.h>

#define STATIC_ASSERT(cond) _Static_assert(cond, #cond)

static void check_types(void) {
    STATIC_ASSERT(sizeof(inferlet_error_code) == sizeof(int));
    STATIC_ASSERT(sizeof(inferlet_chat_history) == 3 * sizeof(const char *));
    STATIC_ASSERT(offsetof(inferlet_chat_history, system) == 0);
    STATIC_ASSERT(offsetof(inferlet_chat_history, query) == sizeof(const char *));
    STATIC_ASSERT(offsetof(inferlet_chat_history, response) == 2 * sizeof(const char *));

    STATIC_ASSERT(offsetof(inferlet_multimodal, prompt_text) == 0);
    STATIC_ASSERT(offsetof(inferlet_multimodal, feature) == sizeof(const char *));
    STATIC_ASSERT(offsetof(inferlet_multimodal, reserved_p) == 2 * sizeof(void *));
    STATIC_ASSERT(offsetof(inferlet_multimodal, size) == 4 * sizeof(void *));
    STATIC_ASSERT(offsetof(inferlet_multimodal, reserved_i) == 4 * sizeof(void *) + sizeof(int));

    STATIC_ASSERT(offsetof(inferlet_multimodal_chat_content_ext, frame_index) == 0);
    STATIC_ASSERT(offsetof(inferlet_multimodal_chat_content_ext, language) == sizeof(int *));

    STATIC_ASSERT(offsetof(inferlet_multimodal_chat_content, type) == 0);
    STATIC_ASSERT(offsetof(inferlet_multimodal_chat_content, data) == sizeof(const char *));

    STATIC_ASSERT(offsetof(inferlet_multimodal_chat_history, role) == 0);
    STATIC_ASSERT(offsetof(inferlet_multimodal_chat_history, content) == sizeof(const char *));
}

static void check_function_declarations(void) {
    void (*p_init)(void) = inferlet_init;
    void (*p_deinit)(void) = inferlet_deinit;
    void *(*p_load)(const char *, void *) = inferlet_load_from_file;
    void (*p_unload)(void *) = inferlet_unload;
    int (*p_run)(void *, void *, void **) = inferlet_run;
    int (*p_run_async)(void *, void *, void **) = inferlet_run_async;
    int (*p_run_stream)(void *, void *, void **) = inferlet_run_stream;
    int (*p_wait_task)(void *, void *, void **) = inferlet_wait_task;
    int (*p_query_task)(void *, void *, void **) = inferlet_query_task;
    int (*p_cancel_task)(void *, void *) = inferlet_cancel_task;
    int (*p_check_active)(void *, int *) = inferlet_check_backend_active;

    void *(*p_in_alloc)(void) = inferlet_task_input_allocate;
    void (*p_in_dealloc)(void *) = inferlet_task_input_deallocate;
    int (*p_in_prompt)(void *, const char *) = inferlet_task_input_set_prompt_text;
    int (*p_in_tokens)(void *, int *, int) = inferlet_task_input_set_prompt_tokens;
    int (*p_in_chat)(void *, inferlet_chat_history[], int) = inferlet_task_input_set_chat_history;
    int (*p_in_mm)(void *, inferlet_multimodal[], int) = inferlet_task_input_set_multimodal;
    int (*p_in_mm_chat)(void *, inferlet_multimodal_chat_history[], int) = inferlet_task_input_set_multimodal_chat_history;
    int (*p_in_lora)(void *, int) = inferlet_task_input_set_low_rank_adapter_id;
    int (*p_in_beam)(void *, int) = inferlet_task_input_set_beam_size;
    int (*p_in_topk)(void *, int) = inferlet_task_input_set_top_k;
    int (*p_in_topp)(void *, float) = inferlet_task_input_set_top_p;
    int (*p_in_temp)(void *, float) = inferlet_task_input_set_temperature;
    int (*p_in_rep)(void *, float) = inferlet_task_input_set_repetition_penalty;
    int (*p_in_maxtok)(void *, int) = inferlet_task_input_set_max_output_tokens;
    int (*p_in_prob)(void *, int) = inferlet_task_input_set_probability_type;
    int (*p_in_verb)(void *, int) = inferlet_task_input_set_verbose_data_flag;
    int (*p_in_block)(void *, int) = inferlet_task_input_set_stream_block_mode;
    int (*p_in_think)(void *, int) = inferlet_task_input_set_thinking_mode;
    int (*p_in_spec)(void *, int) = inferlet_task_input_set_special_token_mode;
    int (*p_in_choices)(void *, const char **, int) = inferlet_task_input_set_guided_choices;
    int (*p_in_tools)(void *, const char **, int) = inferlet_task_input_set_tools;
    int (*p_in_vis)(void *, const char *) = inferlet_task_input_set_visual_encoder;

    void (*p_out_dealloc)(void *) = inferlet_task_output_deallocate;
    const char *(*p_out_text)(void *) = inferlet_task_output_get_result_text;
    const int *(*p_out_tokens)(void *, int *) = inferlet_task_output_get_result_tokens;
    const int *(*p_out_offsets)(void *) = inferlet_task_output_get_token_offsets;
    const float *(*p_out_prob)(void *) = inferlet_task_output_get_token_probability;
    const int *(*p_out_inp_tok)(void *, int *, const char ***) = inferlet_task_output_get_input_tokens;
    const float **(*p_out_logits)(void *, int *, int *) = inferlet_task_output_get_output_logits;
    const float **(*p_out_embed)(void *, int *, int *) = inferlet_task_output_get_output_embedding;

    const char *(*p_str_text)(void *, void *) = inferlet_get_stream_text;
    const int *(*p_str_tokens)(void *, int *) = inferlet_get_stream_tokens;
    const int *(*p_str_offsets)(void *) = inferlet_get_stream_token_offsets;
    const float *(*p_str_prob)(void *) = inferlet_get_stream_token_probability;
    const float **(*p_str_logits)(void *, int *, int *) = inferlet_get_stream_output_logits;
    const float **(*p_str_embed)(void *, int *, int *) = inferlet_get_stream_output_embedding;
    const int *(*p_str_inp_tok)(void *, int *, const char ***) = inferlet_get_stream_input_tokens;

    void *(*p_param_alloc)(void) = inferlet_parameter_allocate;
    void (*p_param_dealloc)(void *) = inferlet_parameter_deallocate;
    void (*p_param_tok_only)(void *, int) = inferlet_parameter_set_tokenizer_only;
    void (*p_param_tok_slow)(void *, int) = inferlet_parameter_set_tokenizer_slow;
    void (*p_param_vis_patch)(void *, int) = inferlet_parameter_set_vision_max_patch_cnt;
    void (*p_param_vis_grid)(void *, int) = inferlet_parameter_set_vision_max_grid_size;
    void (*p_param_vis_skip)(void *, int) = inferlet_parameter_set_vision_trans_skipped;
    void (*p_param_dec_max)(void *, int) = inferlet_parameter_set_decoder_max_token_cnt;
    void (*p_param_dev_id)(void *, int) = inferlet_parameter_set_device_id;
    void (*p_param_dev_list)(void *, int *, int) = inferlet_parameter_set_device_list;
    void (*p_param_backend_cnt)(void *, int) = inferlet_parameter_set_backend_cnt;
    void (*p_param_lora)(void *, int, const char *) = inferlet_parameter_set_low_rank_adapter;
    void (*p_param_load_sync)(void *, int) = inferlet_parameter_set_load_from_file_sync;
    void (*p_param_run_cfg)(void *, const char *) = inferlet_parameter_set_run_config_file;

    int (*p_tok_encode)(void *, const char *, int, int *, int, int *, int, int) = inferlet_tokenizer_encode;
    int (*p_tok_decode)(void *, const int *, int, char *, int, int *, int) = inferlet_tokenizer_decode;

    (void)p_init; (void)p_deinit; (void)p_load; (void)p_unload;
    (void)p_run; (void)p_run_async; (void)p_run_stream; (void)p_wait_task;
    (void)p_query_task; (void)p_cancel_task; (void)p_check_active;
    (void)p_in_alloc; (void)p_in_dealloc; (void)p_in_prompt; (void)p_in_tokens;
    (void)p_in_chat; (void)p_in_mm; (void)p_in_mm_chat; (void)p_in_lora;
    (void)p_in_beam; (void)p_in_topk; (void)p_in_topp; (void)p_in_temp;
    (void)p_in_rep; (void)p_in_maxtok; (void)p_in_prob; (void)p_in_verb;
    (void)p_in_block; (void)p_in_think; (void)p_in_spec; (void)p_in_choices;
    (void)p_in_tools; (void)p_in_vis;
    (void)p_out_dealloc; (void)p_out_text; (void)p_out_tokens; (void)p_out_offsets;
    (void)p_out_prob; (void)p_out_inp_tok; (void)p_out_logits; (void)p_out_embed;
    (void)p_str_text; (void)p_str_tokens; (void)p_str_offsets; (void)p_str_prob;
    (void)p_str_logits; (void)p_str_embed; (void)p_str_inp_tok;
    (void)p_param_alloc; (void)p_param_dealloc; (void)p_param_tok_only;
    (void)p_param_tok_slow; (void)p_param_vis_patch; (void)p_param_vis_grid;
    (void)p_param_vis_skip; (void)p_param_dec_max; (void)p_param_dev_id;
    (void)p_param_dev_list; (void)p_param_backend_cnt; (void)p_param_lora;
    (void)p_param_load_sync; (void)p_param_run_cfg;
    (void)p_tok_encode; (void)p_tok_decode;
}

int main(void) {
    check_types();
    check_function_declarations();
    return 0;
}

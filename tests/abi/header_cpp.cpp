#include "inferlet.h"
#include <cstddef>
#include <type_traits>

static_assert(sizeof(inferlet_error_code) == sizeof(int), "ErrorCode size mismatch");
static_assert(sizeof(inferlet_chat_history) == 3 * sizeof(const char *), "ChatHistory size mismatch");
static_assert(offsetof(inferlet_chat_history, system) == 0, "ChatHistory system offset mismatch");
static_assert(offsetof(inferlet_chat_history, query) == sizeof(const char *), "ChatHistory query offset mismatch");
static_assert(offsetof(inferlet_chat_history, response) == 2 * sizeof(const char *), "ChatHistory response offset mismatch");

static_assert(offsetof(inferlet_multimodal, prompt_text) == 0, "MultiModal prompt_text offset mismatch");
static_assert(offsetof(inferlet_multimodal, feature) == sizeof(const char *), "MultiModal feature offset mismatch");
static_assert(offsetof(inferlet_multimodal, reserved_p) == 2 * sizeof(void *), "MultiModal reserved_p offset mismatch");
static_assert(offsetof(inferlet_multimodal, size) == 4 * sizeof(void *), "MultiModal size offset mismatch");
static_assert(offsetof(inferlet_multimodal, reserved_i) == 4 * sizeof(void *) + sizeof(int), "MultiModal reserved_i offset mismatch");

static_assert(offsetof(inferlet_multimodal_chat_content_ext, frame_index) == 0, "MultiModal_ChatContentExt frame_index offset mismatch");
static_assert(offsetof(inferlet_multimodal_chat_content_ext, language) == sizeof(int *), "MultiModal_ChatContentExt language offset mismatch");

static_assert(offsetof(inferlet_multimodal_chat_content, type) == 0, "MultiModal_ChatContent type offset mismatch");
static_assert(offsetof(inferlet_multimodal_chat_content, data) == sizeof(const char *), "MultiModal_ChatContent data offset mismatch");

static_assert(offsetof(inferlet_multimodal_chat_history, role) == 0, "MultiModal_ChatHistory role offset mismatch");
static_assert(offsetof(inferlet_multimodal_chat_history, content) == sizeof(const char *), "MultiModal_ChatHistory content offset mismatch");

void check_function_declarations_cpp() {
    auto p_init = inferlet_init;
    auto p_deinit = inferlet_deinit;
    auto p_load = inferlet_load_from_file;
    auto p_unload = inferlet_unload;
    auto p_run = inferlet_run;
    auto p_run_async = inferlet_run_async;
    auto p_run_stream = inferlet_run_stream;
    auto p_wait_task = inferlet_wait_task;
    auto p_query_task = inferlet_query_task;
    auto p_cancel_task = inferlet_cancel_task;
    auto p_check_active = inferlet_check_backend_active;

    auto p_in_alloc = inferlet_task_input_allocate;
    auto p_in_dealloc = inferlet_task_input_deallocate;
    auto p_in_prompt = inferlet_task_input_set_prompt_text;
    auto p_in_tokens = inferlet_task_input_set_prompt_tokens;
    auto p_in_chat = inferlet_task_input_set_chat_history;
    auto p_in_mm = inferlet_task_input_set_multimodal;
    auto p_in_mm_chat = inferlet_task_input_set_multimodal_chat_history;
    auto p_in_lora = inferlet_task_input_set_low_rank_adapter_id;
    auto p_in_beam = inferlet_task_input_set_beam_size;
    auto p_in_topk = inferlet_task_input_set_top_k;
    auto p_in_topp = inferlet_task_input_set_top_p;
    auto p_in_temp = inferlet_task_input_set_temperature;
    auto p_in_rep = inferlet_task_input_set_repetition_penalty;
    auto p_in_maxtok = inferlet_task_input_set_max_output_tokens;
    auto p_in_prob = inferlet_task_input_set_probability_type;
    auto p_in_verb = inferlet_task_input_set_verbose_data_flag;
    auto p_in_block = inferlet_task_input_set_stream_block_mode;
    auto p_in_think = inferlet_task_input_set_thinking_mode;
    auto p_in_spec = inferlet_task_input_set_special_token_mode;
    auto p_in_choices = inferlet_task_input_set_guided_choices;
    auto p_in_tools = inferlet_task_input_set_tools;
    auto p_in_vis = inferlet_task_input_set_visual_encoder;

    auto p_out_dealloc = inferlet_task_output_deallocate;
    auto p_out_text = inferlet_task_output_get_result_text;
    auto p_out_tokens = inferlet_task_output_get_result_tokens;
    auto p_out_offsets = inferlet_task_output_get_token_offsets;
    auto p_out_prob = inferlet_task_output_get_token_probability;
    auto p_out_inp_tok = inferlet_task_output_get_input_tokens;
    auto p_out_logits = inferlet_task_output_get_output_logits;
    auto p_out_embed = inferlet_task_output_get_output_embedding;

    auto p_str_text = inferlet_get_stream_text;
    auto p_str_tokens = inferlet_get_stream_tokens;
    auto p_str_offsets = inferlet_get_stream_token_offsets;
    auto p_str_prob = inferlet_get_stream_token_probability;
    auto p_str_logits = inferlet_get_stream_output_logits;
    auto p_str_embed = inferlet_get_stream_output_embedding;
    auto p_str_inp_tok = inferlet_get_stream_input_tokens;

    auto p_param_alloc = inferlet_parameter_allocate;
    auto p_param_dealloc = inferlet_parameter_deallocate;
    auto p_param_tok_only = inferlet_parameter_set_tokenizer_only;
    auto p_param_tok_slow = inferlet_parameter_set_tokenizer_slow;
    auto p_param_vis_patch = inferlet_parameter_set_vision_max_patch_cnt;
    auto p_param_vis_grid = inferlet_parameter_set_vision_max_grid_size;
    auto p_param_vis_skip = inferlet_parameter_set_vision_trans_skipped;
    auto p_param_dec_max = inferlet_parameter_set_decoder_max_token_cnt;
    auto p_param_dev_id = inferlet_parameter_set_device_id;
    auto p_param_dev_list = inferlet_parameter_set_device_list;
    auto p_param_backend_cnt = inferlet_parameter_set_backend_cnt;
    auto p_param_lora = inferlet_parameter_set_low_rank_adapter;
    auto p_param_load_sync = inferlet_parameter_set_load_from_file_sync;
    auto p_param_run_cfg = inferlet_parameter_set_run_config_file;

    auto p_tok_encode = inferlet_tokenizer_encode;
    auto p_tok_decode = inferlet_tokenizer_decode;

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

int main() {
    check_function_declarations_cpp();
    return 0;
}

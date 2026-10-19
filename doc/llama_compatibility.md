# llama.cpp Compatibility Matrix and API Mapping

This document specifies the compatibility details and API mapping between `Inferlet` and the pinned `llama.cpp` upstream commit.

## 1. Pinned Commit

- **Commit SHA**: `8b4b3558f1459c13e4aa38d5c94d306a00dc6acd`
- **Subproject Path**: `third_party/llama.cpp`
- **Primary Header**: `include/llama.h`
- **GGML Headers**: `ggml/include/ggml.h`, `ggml/include/ggml-cpu.h`

## 2. API Mapping Table

| Inferlet Feature | llama.cpp API / Concepts | Notes |
| :--- | :--- | :--- |
| Global Backend Init | `llama_backend_init()` | Called once on initial runtime initialization |
| Global Backend Free | `llama_backend_free()` | Called once when runtime reference count drops to zero |
| Model Default Params | `llama_model_default_params()` | Provides base struct for model loading |
| Model Load | `llama_model_load_from_file(path, params)` | Loads GGUF model weights |
| Model Free | `llama_model_free(model)` | RAII handled in `LlamaEngine` |
| Context Default Params | `llama_context_default_params()` | Base struct for per-task execution context |
| Context Init | `llama_init_from_model(model, params)` | Creates execution context |
| Context Free | `llama_free(ctx)` | Per-task cleanup |
| Vocab Access | `llama_model_get_vocab(model)` | Returns pointer to `llama_vocab` |
| Vocab Token Count | `llama_vocab_n_tokens(vocab)` | Number of tokens in vocabulary |
| EOG Token Check | `llama_vocab_is_eog(vocab, token)` | Identifies end-of-generation tokens |
| Token to Piece | `llama_token_to_piece(vocab, token, buf, len, lstrip, special)` | Extracts text slice for token |
| Tokenize | `llama_tokenize(vocab, text, text_len, tokens, max_tokens, add_special, parse_special)` | Encodes string to tokens |
| Chat Template | `llama_model_chat_template(model, name)`, `llama_chat_apply_template(tmpl, chat, n_msg, add_ass, buf, len)` | Applies Jinja chat template |
| Decode Execution | `llama_decode(ctx, batch)` | Submits batch for prompt processing or token generation |
| Logits Access | `llama_get_logits_ith(ctx, i)` | Obtains output logits |
| Embeddings Access | `llama_get_embeddings_seq(ctx, seq)` / `llama_get_embeddings_ith(ctx, i)` | Enabled when embeddings flag is requested |
| Sampler Greedy | `llama_sampler_init_greedy()` | Greedy token selection |
| Sampler Distribution | `llama_sampler_init_dist(seed)` | Seeded random distribution |
| Sampler Top-K | `llama_sampler_init_top_k(k)` | Top-K candidate filter |
| Sampler Top-P | `llama_sampler_init_top_p(p, min_keep)` | Top-P nucleus filtering |
| Sampler Temp | `llama_sampler_init_temp(t)` | Temperature scaling |
| Sampler Penalties | `llama_sampler_init_penalties(n_vocab, eos, nl, last_n, repeat, freq, present, penalize_nl, ignore_eos)` | Repetition penalties |
| Sampler Grammar | `llama_sampler_init_grammar(vocab, grammar_str, grammar_root)` | Guided choice / structured grammar constraint |
| LoRA Adapter | `llama_adapter_lora_init(path)`, `llama_set_adapters_lora(adapters, n, ...)` | LoRA runtime application |

## Image integration

`vision.mmproj` loads `mtmd_init_from_file`; CHW UINT8 images are converted to RGB bitmaps and processed by `mtmd_tokenize` and `mtmd_encode_chunk`. `mtmd_helper_decode_image_chunk` handles image embeddings and model-specific positions/attention; text chunks use `mtmd_helper_eval_chunk_single`. See [vision.md](vision.md) for tested models, input layout and unsupported combinations. The adapter is isolated in `src/multimodal/mtmd_backend.*`.

## 3. ABI Boundary Safety Guarantees

1. `sizeof(llama_token) == sizeof(int)` is verified at compile time with `static_assert`.
2. No llama.cpp or ggml types or headers are exposed in `include/inferlet.h`.
3. All llama.cpp resources (`llama_model`, `llama_context`, `llama_sampler`, etc.) are owned by unique RAII smart pointers inside `LlamaEngine`.

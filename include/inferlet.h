#ifndef INFERLET_H_INCLUDED
#define INFERLET_H_INCLUDED

#if defined(_WIN32) || defined(__CYGWIN__)
#  if defined(INFERLET_BUILDING_LIBRARY) && defined(INFERLET_SHARED)
#    define INFERLET_API __declspec(dllexport)
#  elif defined(INFERLET_SHARED)
#    define INFERLET_API __declspec(dllimport)
#  else
#    define INFERLET_API
#  endif
#else
#  if defined(INFERLET_BUILDING_LIBRARY) && defined(INFERLET_SHARED)
#    define INFERLET_API __attribute__((visibility("default")))
#  else
#    define INFERLET_API
#  endif
#endif

typedef enum inferlet_error_code
{
    INFERLET_OK                  =  0,
    INFERLET_E_INVALID_ARGUMENT  = -1,
    INFERLET_E_INVALID_HANDLE    = -2,
    INFERLET_E_INVALID_STATE     = -3,
    INFERLET_E_IO                = -4,
    INFERLET_E_MODEL_LOAD        = -5,
    INFERLET_E_INFERENCE         = -6,
    INFERLET_E_UNSUPPORTED       = -7,
    INFERLET_E_CANCELLED         = -8,
    INFERLET_E_BUFFER_TOO_SMALL  = -9,
    INFERLET_E_OUT_OF_MEMORY     = -10,
    INFERLET_E_INTERNAL          = -11
} inferlet_error_code;

#ifdef __cplusplus
extern "C"
{
#endif

    // *************参数名称说明***********
    // task_in  :   任务输入
    // task_out :   任务输出
    // task     :   正在执行的非流式任务
    // task_stream: 正在执行的流式任务

    // *************函数族***********
    // task_in  :  - 内容
    //              inferlet_task_input_set_prompt_text
    //              inferlet_task_input_set_prompt_tokens
    //              inferlet_task_input_set_chat_history
    //              inferlet_task_input_set_multimodal
    //             - 后处理
    //              inferlet_task_input_set_beam_size
    //              inferlet_task_input_set_top_k
    //              inferlet_task_input_set_top_p
    //              inferlet_task_input_set_temperature
    //              inferlet_task_input_set_repetition_penalty
    //              inferlet_task_input_set_probability_type
    //             - 管理
    //              inferlet_task_input_allocate
    //              inferlet_task_input_deallocate
    // task_out :
    //              inferlet_task_output_deallocate
    //              inferlet_task_output_get_result_text
    //              inferlet_task_output_get_result_tokens
    //              inferlet_task_output_get_token_offsets
    //              inferlet_task_output_get_token_probability
    // task_stream:
    //              inferlet_get_stream_text
    //              inferlet_get_stream_tokens
    //              inferlet_get_stream_token_offsets
    //              inferlet_get_stream_token_probability

    // *******************************************流程说明************************************
    //                               inferlet_init(模块初始化)
    //                                     |
    //                         inferlet_parameter_allocate(申请参数(可选))  --------> 设置参数(接口待按需开发)
    //                                     |
    //                            inferlet_load_from_file(加载模型)
    //                                     |
    //                         inferlet_task_input_allocate(申请任务输入)    --------> 设置具体输入信息、后处理信息等
    //                                     |
    //           ———————————————————————————————————————————————————
    //          |                          |                        |
    // inferlet_run_async(异步)        inferlet_run(同步)     inferlet_run_stream(流式)
    //          |                          |                        |
    // inferlet_query_task                   |                        |
    // inferlet_wait_task                    |                        |
    // inferlet_cancel_task                  |                        |
    //          |                          |                        |
    //           ———————————————————————————                        |
    //                         |                                    |
    //          inferlet_task_output_get_result_text         inferlet_get_stream_text <--  ---->获取输出对应的token\偏移\置信度
    //                         |                                      |          |
    //          inferlet_task_output_deallocate(释放任务输出)            |-----------
    //                         |                                      |
    //                          ——————————————————————————————————————
    //                                      |
    //                           inferlet_task_input_deallocate(释放任务输入)
    //                                      |
    //                              inferlet_unload(卸载模型)
    //                                      |
    //                               inferlet_deinit(模块反初始化)

    // 对话历史记录，当对应信息不存在时，可设置为空指针或空字符串.
    typedef struct inferlet_chat_history
    {
        const char *system;   //系统提示
        const char *query;    //查询文本
        const char *response; //响应文本;如果是连续对话的当前轮查询,则该response为空
    } inferlet_chat_history;

    // 多模态数据，当对应信息不存在时，可设置为空指针或空字符串.
    // 多模态数据顺序按照结构体成员先后位置拼接,譬如文本+视觉,不支持视觉+文本
    typedef struct inferlet_multimodal
    {
        const char *prompt_text; // 位于特征数据前面的提示语.
        const void *feature;     // 多模态特征数据指针，目前实现仅支持 Host + FP32 数据.
        void *reserved_p[2];     // 保留.
        int size;                // 特征的数据量, 字节为单位. size = n_feature * n_embd * element_size
        int reserved_i[7];       // 保留.
    } inferlet_multimodal;

    // 多模态对话内容扩展信息.
    typedef struct inferlet_multimodal_chat_content_ext
    {
        int *frame_index;       // 帧索引，若不提供则默认为从零开始的连续帧
        const char *language;   // audio 对应的语言, 与模型相关.
        void *reserved_p[6];    // 保留.
        int frame_num;          // 帧数量，即帧索引数组的长度.
        int packing_num;        // 视频帧打包数量. 有效范围 1-6
        int slice_num;          // 切分子图的网格行数, 非切分子图不填
        int reserved_i[9];      // 保留.
    } inferlet_multimodal_chat_content_ext;

    // 多模态对话内容.
    // 如果 type 为 "image" 或 "video",
    // 当数据类型为 float/half/bf16 时，data 为特征数据，是按照 patch 顺序排列存放的。
    //   特征数据的数量为 n_feature = grid_t * grid_h * grid_w, 数据量 size = n_feature * n_embd * element_size
    // 当数据类型为 uint8 时，data 为图像数据，按照 [t, c, h, w] 的顺序进行存储，数据量 size = t * c * h * w
    //   其中 t 为图像或视频的帧数，必须为 1 (单帧)或者 grid_t * temporal_patch_size (多帧).
    //   c 为与模型相关的固定值，h = grid_h * patch_size, w = grid_w * patch_size.
    // 当前 llama.cpp 后端支持 type="image", data_type=4, grid_t=1, c=3 的 RGB 平面数据。
    // 需在运行 JSON 中配置 vision.mmproj；patch_size 取该投影文件的 clip.vision.patch_size。
    // JPEG/PNG 应先解码为 RGB，再由 HWC 转为 CHW；不能直接传入压缩文件字节。
    // 视频、预计算特征和非默认 ext 暂不支持。详见 doc/vision.md。
    typedef struct inferlet_multimodal_chat_content
    {
        const char *type;        // "text", "image", "video" 等
        const void *data;        // 如果 type 为 "text", 则 data 为字符串(const char *)
                                 // 如果 type 为 "image" 或 "video", 则 data 为特征数据或图像数据.
        const char *cache_key;   // 缓存 key, 用于存储或查询多模态图像或视频特征数据，减少重复计算.
                                 // 在 type 为 "image" 或 "video" 时有效. key 需保证唯一性，否则行为未定义.
                                 // 若为空指针或空字符串，表示不对该图像或视频特征数据使用缓存功能.
        inferlet_multimodal_chat_content_ext *ext;    // 扩展信息，为 null 时表示扩展信息中所有字段均为 null 或 0.
        int size;                // 特征的数据量, 字节为单位.
        int grid_t;              // 时间方向 patch 数
        int grid_h;              // 垂直方向 patch 数
        int grid_w;              // 水平方向 patch 数
        int data_type;           // 特征数据类型. 0: undef, 1: float, 2: half, 3: bf16. 4: uint8, 同一 Task 需保持一致.
        float second_per_grid_t; // 视频帧抽取时间间隔, 可不配置
        int cache_times;         // 缓存使用次数,  相同 key 应设置相同的值, 正常使用应不小于 2.
                                 // 对应 key 的特征数据使用量达到指定次数后, 将在缓存池中自动释放.
        float fps;               // 帧率，用于计算每帧的时间戳. (默认值为 24)
    } inferlet_multimodal_chat_content;

    // 多模态对话历史记录.
    typedef struct inferlet_multimodal_chat_history
    {
        const char *role;                        // 对话角色, "system", "user", "assistant" 等
        inferlet_multimodal_chat_content *content; // 对话内容
        int length;                              // 对话内容的数量.
        int reserved_i;                          // 保留
    } inferlet_multimodal_chat_history;

    /**
 * @brief 模块初始化
 *
 * @param [out]
 * @param [in]
 *
 * @return
 */
    INFERLET_API void inferlet_init(void);

    /**
 * @brief 模块反初始化
 *
 * @param [out]
 * @param [in]
 *
 * @return
 *
 * @note 只有当发起inferlet_init的所有用户全部执行Deinit后,内部才会进入销毁调度相关的管理信息
 */
    INFERLET_API void inferlet_deinit(void);

    /**
 * @brief 模型加载
 *
 * @param [out]
 * @param [in] file_name        文件路径
 * @param [in] param            通过inferlet_parameter_allocate申请的参数信息
 *
 * @return                      模型句柄,非空=创建成功;NULL=创建失败
 * @note handle 不是线程安全的，不能在多个线程中同时使用同一个 handle.
 *       可以在同一个线程中通过 inferlet_run_async, inferlet_run_stream 在同一个 handle 上同时运行多个任务.
 *       如果需要在多个线程中同时使用，需要创建多个 handle, 后台推理引擎会共享数据.
 *       对于同一个模型，由于后台数据共享，仅第一次加载模型时提供的参数有效。
 *       如果同一个模型多次加载时的参数不一致，可能行为与预期的会存在差异。
 */
    INFERLET_API void *inferlet_load_from_file(const char *file_name, void *param);

    /**
 * @brief 模型卸载
 *
 * @param [out]
 * @param [in]  handle          模型句柄
 *
 * @return                      模型句柄,非空=创建成功;NULL=创建失败
 * @note 该接口需和inferlet_load_from_file配套使用
 */
    INFERLET_API void inferlet_unload(void *handle);

    /**
 * @brief 同步接口，运行推理
 *
 * @param [out] task_out        任务输出; 当返回值为非负值时有效
 * @param [in]  handle          模型句柄
 * @param [in]  task_in         Alloc分配的任务输入指针
 *
 * @return                      0=运行成功;负值=运行失败
 * @note 该接口需和inferlet_load_from_file配套使用
 *       生成结束时若输出文本不是完整有效的 UTF-8，返回 INFERLET_E_INFERENCE。
 */
    INFERLET_API int inferlet_run(void *handle, void *task_in, void **task_out);

    /**
 * @brief 异步接口，运行推理
 *
 * @param [out] task            正在执行的任务句柄;当返回值为非负值时有效
 * @param [in]  handle          模型句柄
 * @param [in]  task_in         Alloc分配的任务输入指针
 *
 * @return                      0=运行成功;负值=运行失败
 * @note 该接口可与inferlet_query_task、inferlet_wait_task配合实现异步推理
 */
    INFERLET_API int inferlet_run_async(void *handle, void *task_in, void **task);

    /**
 * @brief 流式推理
 *
 * @param [out] task_stream     正在执行的流式推理任务句柄;当返回值为非负值时有效
 * @param [in]  handle          模型句柄
 * @param [in]  task_in         Alloc分配的任务输入
 *
 * @return                      0=运行成功;负值=运行失败
 * @note 该接口与inferlet_get_stream_text配合完成流式推理
 */
    INFERLET_API int inferlet_run_stream(void *handle, void *task_in, void **task_stream);

    /**
 * @brief 等待任务完成
 *
 * @param [out] task_out        任务输出; 当返回值为非负值时有效
 * @param [in]  handle          模型句柄
 * @param [in]  task            正在执行的任务句柄
 *
 * @return                      0=成功;负值=失败
 * @note 同步接口,直到任务完成或异常
 */
    INFERLET_API int inferlet_wait_task(void *handle, void *task, void **task_out);

    /**
 * @brief 查询任务是否完成
 *
 * @param [out] task_out        任务输出;非空时有效
 * @param [in]  handle          模型句柄
 * @param [in]  task            正在执行的任务句柄
 *
 * @return                      0=成功;负值=失败
 * @note 未完成时, *task_out 返回 nullptr, 函数返回 0
 *       出错时,   *task_out 返回 nullptr, 函数返回对应的负错误码
 *       正常完成时, *task_out 返回非空, 函数返回 0
 */
    INFERLET_API int inferlet_query_task(void *handle, void *task, void **task_out);

    /**
 * @brief 取消当前正在执行的任务
 *
 * @param [out]
 * @param [in]  handle          模型句柄
 * @param [in]  task            正在执行的任务句柄
 *
 * @return                      0=取消成功;负值=取消失败
 * @note
 */
    INFERLET_API int inferlet_cancel_task(void *handle, void *task);

    /**
 * @brief 检查推理引擎后台是否处于工作状态
 *
 * @param [out] is_active       返回状态: 1=活跃(是), 0=不活跃(否)
 * @param [in]  handle          模型句柄
 * @return                      0=成功;负值=失败
 * @note
 */
    INFERLET_API int inferlet_check_backend_active(void *handle, int *is_active);

    /**
 * @brief 申请分配任务输入
 *
 * @param [out]
 * @param [in]
 *
 * @return                      非空,返回分配得到的任务输入
 * @note  需和 Deallocate 配套使用,否则存在任务泄漏
 */
    INFERLET_API void *inferlet_task_input_allocate(void);

    /**
 * @brief 释放分配的任务输入
 *
 * @param [out]
 * @param [in] task_in          任务输入
 *
 * @return
 * @note
 */
    INFERLET_API void inferlet_task_input_deallocate(void *task_in);

    /**
 * @brief 为该任务输入设置输入的提示文本
 *
 * @param [out]
 * @param [in] task_in          任务输入
 * @param [in] prompt           提示文本
 *
 * @return                      0=成功;负值=失败
 * @note
 */
    INFERLET_API int inferlet_task_input_set_prompt_text(void *task_in, const char *prompt);

    /**
 * @brief 为该任务输入设置输入的提示token
 *
 * @param [out]
 * @param [in] task_in          任务输入
 * @param [in] tokens           token指针
 * @param [in] length           token个数,非字节数
 *
 * @return                      0=成功;负值=失败
 * @note
 */
    INFERLET_API int inferlet_task_input_set_prompt_tokens(void *task_in, int *tokens, int length);

    /**
 * @brief 为该任务输入设置历史聊天记录
 *
 * @param [out]
 * @param [in] task_in          任务输入
 * @param [in] history          历史记录数组指针
 * @param [in] length           数组成员个数
 *
 * @return                      0=成功;负值=失败
 * @note
 */
    INFERLET_API int inferlet_task_input_set_chat_history(void *task_in, inferlet_chat_history history[], int length);

    /**
 * @brief 为该任务输入设置多模态输入数据
 *
 * @param [out]
 * @param [in] task_in          任务输入
 * @param [in] input            多模态数据数组指针
 * @param [in] length           数组成员个数
 * @return                      0=成功;负值=失败
 * @note 当前后端不支持此 legacy 特征输入；设置成功后执行接口会返回 INFERLET_E_UNSUPPORTED。
 *       原始图像请使用 SetMultiModal_ChatHistory 并配置匹配的 vision.mmproj。
 */
    INFERLET_API int inferlet_task_input_set_multimodal(void *task_in, inferlet_multimodal input[], int length);

    /**
 * @brief 为该任务输入设置多模态聊天数据(VL模型)
 *
 * @param [out]
 * @param [in] task_in          任务输入
 * @param [in] history          多模态历史记录数组指针
 * @param [in] length           数组成员个数
 * @return                      0=成功;负值=失败
 * @note 支持配置匹配 vision.mmproj 的视觉模型，图像需为 UINT8 RGB CHW 数据。
 *       视频、预计算特征及非默认帧/切片扩展不支持；详见 doc/vision.md。
 */
    INFERLET_API int inferlet_task_input_set_multimodal_chat_history(
        void *task_in, inferlet_multimodal_chat_history history[], int length);

    /**
 * @brief 为该任务输入设置lora模型
 *
 * @param [out]
 * @param [in] task_in          任务输入
 * @param [in] lora_id          lora模型编号
 * @return                      0=成功;负值=失败
 * @note
 */
    INFERLET_API int inferlet_task_input_set_low_rank_adapter_id(void *task_in, int lora_id);

    /**
 * @brief 为该任务输入设置后处理beam search超参：候选个数
 *
 * @param [out]
 * @param [in] task_in          任务输入
 * @param [in] beam_size        候选个数
 * @return                      0=成功;负值=失败
 * @note beam_size > 1 时不支持输出 logits/embedding；请求这些详细输出时执行接口返回 INFERLET_E_UNSUPPORTED。
 */
    INFERLET_API int inferlet_task_input_set_beam_size(void *task_in, int beam_size);

    /**
 * @brief 为该任务输入设置后处理topk个数
 *
 * @param [out]
 * @param [in] task_in          任务输入
 * @param [in] top_k            前N个，0=模型默认值
 * @return                      0=成功;负值=失败
 * @note
 */
    INFERLET_API int inferlet_task_input_set_top_k(void *task_in, int top_k);

    /**
 * @brief 为该任务输入设置后处理top累加概率值
 *
 * @param [out]
 * @param [in] task_in          任务输入
 * @param [in] top_p            累加概率值
 *
 * @return                      0=成功;负值=失败
 * @note
 */
    INFERLET_API int inferlet_task_input_set_top_p(void *task_in, float top_p);

    /**
 * @brief 为该任务输入设置后处理温度系数,增加随机性
 *
 * @param [out]
 * @param [in] task_in          任务输入
 * @param [in] temperature      温度系数
 *
 * @return                      0=成功;负值=失败
 * @note
 */
    INFERLET_API int inferlet_task_input_set_temperature(void *task_in, float temperature);

    /**
 * @brief 为该任务输入设置后处理重复惩罚系数
 *
 * @param [out]
 * @param [in] task_in              任务输入
 * @param [in] repetition_penalty   惩罚系数
 *
 * @return                      0=成功;负值=失败
 * @note
 */
    INFERLET_API int inferlet_task_input_set_repetition_penalty(void *task_in, float repetition_penalty);

    /**
 * @brief 为该任务输入设置最大输出token数
 *
 * @param [out]
 * @param [in] task_in              任务输入
 * @param [in] max_tokens           最大输出token数
 *
 * @return                      0=成功;负值=失败
 * @note
 */
    INFERLET_API int inferlet_task_input_set_max_output_tokens(void *task_in, int max_tokens);

    /**
 * @brief 为该任务输入设置后处理返回token的概率类型
 *
 * @param [out]
 * @param [in] task_in              任务输入
 * @param [in] type                 0: 无概率值(默认), 1: sample 调整前的概率值, 2: sample 调整后的概率值.
 *
 * @return                      0=成功;负值=失败
 * @note
 */
    INFERLET_API int inferlet_task_input_set_probability_type(void *task_in, int type);

// 设置任务处理过程中的保留中间数据的标志，中间数据可使用相应的接口获取.
// 掩码形式，对应的位设置为 1，则可以获取相应的中间数据.
#define INFERLET_VERBOSE_F_INPUT_TOKENS (1 << 0)     // 模型输入 tokens 以及对应的  text 信息
#define INFERLET_VERBOSE_F_OUTPUT_LOGITS (1 << 1)    // 输出 tokens 对应的 logits 向量
#define INFERLET_VERBOSE_F_OUTPUT_EMBEDDING (1 << 2) // 输出 tokens 对应的 embedding 向量
#define INFERLET_VERBOSE_F_ALL (-1)                  // 所有中间数据
    INFERLET_API int inferlet_task_input_set_verbose_data_flag(void *task_in, int flag);

    /**
 * @brief 为该任务输入设置是否结果堵塞, 仅影响流模式运行接口 (inferlet_run_stream).
 *
 * @param [out]
 * @param [in] task_in              任务输入
 * @param [in] mode                 0 (默认) 表示流模式获取结果时不阻塞.
                                    1 表示获取结果时阻塞.
 *
 * @return                      0=成功;负值=失败
 * @note 使用阻塞模式时，inferlet_get_stream_text 仍有可能返回空字符串（""), 但不会无限返回空字符串.
 */
    INFERLET_API int inferlet_task_input_set_stream_block_mode(void *task_in, int mode);

    /**
 * @brief 为该任务输入设置思考模式的开关.
 *
 * @param [out]
 * @param [in] task_in              任务输入
 * @param [in] mode                 0 non-thinking mode.
                                    1 (default) thinking mode.
 *
 * @return                      0=成功;负值=失败
 * @note 当前后端不支持关闭 Qwen3 系列的 thinking mode，mode=0 时执行接口返回 INFERLET_E_UNSUPPORTED。
 *       对不支持 thinking 的模型，此配置不生效。
 */
    INFERLET_API int inferlet_task_input_set_thinking_mode(void *task_in, int mode);

    /**
 * @brief 为该任务输入设置special token在text输出中是否保留.
 *
 * @param [out]
 * @param [in] task_in              任务输入
 * @param [in] keep                 0: 不保留（默认）, 1: 保留.
 *
 * @return                      0=成功;负值=失败
 * @note
 */
    INFERLET_API int inferlet_task_input_set_special_token_mode(void *task_in, int keep);

    /**
 * @brief 为该任务输入设置guided_choice选项.
 *
 * @param [out]
 * @param [in] task_in              任务输入
 * @param [in] choices              guided_choice列表.
 * @param [in] num                  guided_choice数量
 *
 * @return                      0=成功;负值=失败
 * @note
 */
    INFERLET_API int inferlet_task_input_set_guided_choices(void *task_in, const char **choices, int num);

    /**
 * @brief 为该任务输入设置tool列表.
 *
 * @param [out]
 * @param [in] task_in              任务输入
 * @param [in] tools                tool列表
 * @param [in] num                  tool数量
 *
 * @return                      0=成功;负值=失败
 * @note 每个输入tool必须为合法json格式
 */
    INFERLET_API int inferlet_task_input_set_tools(void *task_in, const char **tools, int num);

    /**
 * @brief 为该任务输入设置视觉编码器id.
 *
 * @param [out]
 * @param [in] task_in              任务输入
 * @param [in] id                   视觉编码器id, 空字符串走默认编码器
 *
 * @return                      0=成功;负值=失败
 * @note 当前支持空值或 "default"，使用 vision.mmproj 配置的编码器。
 *       视觉推理时其他编码器 ID 返回 INFERLET_E_UNSUPPORTED。
 */
    INFERLET_API int inferlet_task_input_set_visual_encoder(void *task_in, const char *id);

    /**
 * @brief 释放任务输出
 *
 * @param [out]
 * @param [in] task_out             任务输出
 *
 * @return                          0=成功;负值=失败
 * @note
 */
    INFERLET_API void inferlet_task_output_deallocate(void *task_out);

    /**
 * @brief 获取文本输出
 *
 * @param [out]
 * @param [in] task_out             任务输出
 *
 * @return                          任务输出的文本字符串
 * @note  task_out释放前, 返回的文本指针一直有效
 */
    INFERLET_API const char *inferlet_task_output_get_result_text(void *task_out);

    /**
 * @brief 从任务输出中获取token
 *
 * @param [out] length              任务输出的token数量
 * @param [in] task_out             任务输出
 *
 * @return                          任务输出的token指针
 * @note  task_out释放前, 返回token指针一直有效
 */
    INFERLET_API const int *inferlet_task_output_get_result_tokens(void *task_out, int *length);

    /**
 * @brief 获取每个token输出对应的文本子串偏移值
 *
 * @param [out]
 * @param [in] task_out             任务输出
 *
 * @return                          每个输出token对应文本子串的偏移值
 * @note
 */
    INFERLET_API const int *inferlet_task_output_get_token_offsets(void *task_out);

    /**
 * @brief 获取每个token输出对应的概率值
 *
 * @param [out]
 * @param [in] task_out             任务输出
 *
 * @return                          任务输出的token概率值指针
 * @note
 */
    INFERLET_API const float *inferlet_task_output_get_token_probability(void *task_out);

    /**
 * @brief 获取送给模型的输入 token 信息(非流模式, 同步或异步接口)
 *
 * @param [out]length               token数量
 * @param [out]token_text           token对应字符串，len x string
 * @param [in] task_out             任务输出
 *
 * @return                          token序号
 * @note SetVerboseDataFlag接口使能 INFERLET_VERBOSE_F_INPUT_TOKENS
 *       图像占用的位置以 token=-1、token_text="<image>" 表示，不是可用于文本解码的词表 token。
 */
    INFERLET_API const int *inferlet_task_output_get_input_tokens(void *task_out, int *length, const char ***token_text);

    /**
 * @brief 获取输出 tokens 对应的模型输出数据概率值(非流模式，同步或异步接口)
 *
 * @param [out]length               token数量
 * @param [out]n_vocab              token对应vocab长度
 * @param [in] task_out             任务输出
 *
 * @return                          token概率值, len x n_vocab
 * @note SetVerboseDataFlag接口使能 INFERLET_VERBOSE_F_OUTPUT_LOGITS
 */
    INFERLET_API const float **inferlet_task_output_get_output_logits(void *task_out, int *length, int *n_vocab);

    /**
 * @brief 获取输出 tokens 对应的模型输出数据(非流模式，同步或异步接口)
 *
 * @param [out]length               token数量
 * @param [out]embed_size           token对应的embedding长度
 * @param [in] task_out             任务输出
 *
 * @return                          token embed, len x embed_size
 * @note SetVerboseDataFlag接口使能 INFERLET_VERBOSE_F_OUTPUT_EMBEDDING
 */
    INFERLET_API const float **inferlet_task_output_get_output_embedding(void *task_out, int *length, int *embed_size);

    /**
 * @brief 获取流式输出
 *
 * @param [out]
 * @param [in] handle               模型句柄
 * @param [in] task_stream          正在执行的流式任务句柄
 *
 * @return                          非空字符串，表示有效输出，正在处理任务.
 *                                  空字符串(""), 表示当前没有输出，正在处理任务.
 *                                  空指针(NULL), 表示任务已结束（正常完成、失败或取消），同时自动释放 task_stream 的内存.
 * @note 流式输出时,每次获取因为推理速度、获取间隔等原因,每次返回的输出长度不定;
 *       指针指向的内容在下次调用之前保持有效，建议用户深度拷贝结果,下次只能获取接下来新生成的文本
 *       指针无需用户释放
 *       当前公共接口无法通过 NULL 区分正常结束和失败；需要明确的完成错误码时，
 *       使用 Run 或 inferlet_run_async 配合 inferlet_wait_task/inferlet_query_task。
 */
    INFERLET_API const char *inferlet_get_stream_text(void *handle, void *task_stream);

    /**
 * @brief 获取流式token输出
 *
 * @param [out] length              当前输出文本对应的 token 数量
 * @param [in] task_stream          正在执行的流式任务句柄
 *
 * @return                          返回通过 inferlet_get_stream_text 获取当前文本对应的token
 *                                  指针在下次调用 inferlet_get_stream_text 前或者task_stream释放前有效
 * @note 只有 inferlet_get_stream_text 返回非空字符串时该接口才有意义
 *       指针无需用户释放
 */
    INFERLET_API const int *inferlet_get_stream_tokens(void *task_stream, int *length);

    /**
 * @brief 获取流式token输出的文本子串偏移值
 *
 * @param [out]
 * @param [in] handle               模型句柄
 * @param [in] task_stream          正在执行的流式任务句柄
 *
 * @return                          返回通过 inferlet_get_stream_tokens 获取的每个token对应的文本子串偏移值
 *                                  指针在下次调用 inferlet_get_stream_text 前或者task_stream释放前有效
 * @note 只有 inferlet_get_stream_text 返回非空字符串时该接口才有意义
 *       指针无需用户释放
 */
    INFERLET_API const int *inferlet_get_stream_token_offsets(void *task_stream);

    /**
 * @brief 获取流式token输出的概率值
 *
 * @param [out]
 * @param [in] handle               模型句柄
 * @param [in] task_stream          正在执行的流式任务句柄
 *
 * @return                          返回通过 inferlet_get_stream_tokens 获取的每个token对应的概率值
 *                                  指针在下次调用 inferlet_get_stream_text 前或者task_stream释放前有效
 * @note 只有 inferlet_get_stream_text 返回非空字符串时该接口才有意义
 *       指针无需用户释放
 */
    INFERLET_API const float *inferlet_get_stream_token_probability(void *task_stream);

    /**
 * @brief 获取输出 tokens 对应的模型输出数据概率值(流模式)
 *
 * @param [out]length               token数量
 * @param [out]n_vocab              token对应vocab长度
 * @param [in] task_stream          正在执行的流式任务句柄
 *
 * @return                          token logits, len x n_vocab
 * @note
 */
    INFERLET_API const float **inferlet_get_stream_output_logits(void *task_stream, int *length, int *n_vocab);

    /**
 * @brief 获取输出 tokens 对应的模型输出数据(流模式)
 *
 * @param [out]length               token数量
 * @param [out]embed_size           token对应的embedding长度
 * @param [in] task_stream          正在执行的流式任务句柄
 *
 * @return                          token embed, len x embed_size
 * @note
 */
    INFERLET_API const float **inferlet_get_stream_output_embedding(void *task_stream, int *length, int *embed_size);

    /**
 * @brief 获取送给模型的输入 token 信息(流模式)
 *
 * @param [out]length               token数量
 * @param [out]token_text           token对应字符串，len x string
 * @param [in] task_stream          正在执行的流式任务句柄
 *
 * @return                          token序号
 * @note
 */
    INFERLET_API const int *inferlet_get_stream_input_tokens(void *task_stream, int *length, const char ***token_text);

    /**
 * @brief 申请分配用于模型加载时的参数配置, 和 Deallocate 配套使用
 *
 * @param [out]
 * @param [in]
 *
 * @return                          可用于模型加载接口传入的参数param，非空表示成功
 * @note 通过接口形式该非结构体可以应对未来模型加载需增加的各种可能参数配置
 *       对于同一个模型，由于后台数据共享，仅第一次加载模型时提供的参数有效。
 *       如果同一个模型多次加载时的参数不一致，可能行为与预期的会存在差异。
 */
    INFERLET_API void *inferlet_parameter_allocate(void);

    /**
 * @brief 释放用于模型加载时的参数配置
 *
 * @param [out]
 * @param [in]  param                Allocate返回的参数指针
 *
 * @return
 * @note
 *
 */
    INFERLET_API void inferlet_parameter_deallocate(void *param);

    /**
 * @brief 设置是否只有tokenizer功能,其他功能关闭
 *
 * @param [out]
 * @param [in] param                通过Allocate申请的参数指针
 * @param [in] value                配置是否只有tokenizer功能;
 *                                  默认全功能；
 * @return
 * @note 通过接口形式该非结构体可以应对未来模型加载需增加的各种可能参数配置
 *       对于同一个模型，由于后台数据共享，仅第一次加载模型时提供的参数有效。
 *       如果同一个模型多次加载时的参数不一致，可能行为与预期的会存在差异。
 */
    INFERLET_API void inferlet_parameter_set_tokenizer_only(void *param, int value);

    /**
 * @brief 设置是否使用 slow 兼容模式
 *
 * @param [out]
 * @param [in] param                通过Allocate申请的参数指针
 * @param [in] value                value=0: 不使用 slow 模式, value=1: 使用 slow 模式.
 *                                  默认不使用兼容模式；
 * @return
 * @note 通过接口形式该非结构体可以应对未来模型加载需增加的各种可能参数配置
 *       对于同一个模型，由于后台数据共享，仅第一次加载模型时提供的参数有效。
 *       如果同一个模型多次加载时的参数不一致，可能行为与预期的会存在差异。
 */
    // 当前后端不支持 value=1；使用该配置加载模型会返回 NULL。
    INFERLET_API void inferlet_parameter_set_tokenizer_slow(void *param, int value);

    /**
 * @brief 配置vision 网络单次处理的最大 patch 数量.
 *
 * @param [out]
 * @param [in] param                通过Allocate申请的参数指针
 * @param [in] max_cnt              最大patch数量
 * @return
 * @note 若未配置,则为默认值
 */
    // 当前使用投影器自身的预处理规则，不支持自定义 patch 限制；非零配置会导致加载模型返回 NULL。
    INFERLET_API void inferlet_parameter_set_vision_max_patch_cnt(void *param, int max_cnt);

    /**
 * @brief 配置vision 网络单个方向(x或y)最大网格数量.
 *
 * @param [out]
 * @param [in] param                通过Allocate申请的参数指针
 * @param [in] max_size             grid水平/垂直方向数量
 * @return
 * @note 若未配置,则为默认值
 */
    // 当前使用投影器自身的预处理规则，不支持自定义 grid 限制；非零配置会导致加载模型返回 NULL。
    INFERLET_API void inferlet_parameter_set_vision_max_grid_size(void *param, int max_size);

    /**
 * @brief 配置多模态模型是否跳过视觉编码
 *
 * @param [out]
 * @param [in] param                通过Allocate申请的参数指针
 * @param [in] max_size             grid水平/垂直方向数量
 * @return
 * @note vision 网络跳过处理，当输入视频和图像数据时，直接传输 vision 网络的输出数据作为输入.
 */
    // 当前仅支持原始图像编码，不支持预计算视觉特征；value=1 会导致加载模型返回 NULL。
    INFERLET_API void inferlet_parameter_set_vision_trans_skipped(void *param, int value);

    /**
 * @brief 配置解码网络单次处理的最大 token 数量.
 *
 * @param [out]
 * @param [in] param                通过Allocate申请的参数指针
 * @param [in] max_cnt              解码单次处理的最大 token 数量.
 * @return
 * @note
 */
    INFERLET_API void inferlet_parameter_set_decoder_max_token_cnt(void *param, int max_cnt);

    /**
 * @brief 设置 device id.
 *
 * @param [out]
 * @param [in] param                通过Allocate申请的参数指针
 * @param [in] device_id            指定device_id
 * @return
 * @note
 */
    INFERLET_API void inferlet_parameter_set_device_id(void *param, int device_id);

    /**
 * @brief 设置多个 device 的列表.
 *
 * @param [out]
 * @param [in] param                通过Allocate申请的参数指针
 * @param [in] device_list          device id列表
 * @param [in] device_cnt           device数量
 * @return
 * @note 当 device_list 为空时, 使用默认的列表 = [0, 1, ..., device_cnt - 1]
 */
    INFERLET_API void inferlet_parameter_set_device_list(void *param, int *device_list, int device_cnt);

    /**
 * @brief 设置后台处理的引擎数量.
 *
 * @param [out]
 * @param [in] param                通过Allocate申请的参数指针
 * @param [in] backend_cnt          引擎数量
 * @return
 * @note 要求 device_cnt / backend_cnt 为 2 的幂次
 */
    INFERLET_API void inferlet_parameter_set_backend_cnt(void *param, int backend_cnt);

    /**
 * @brief 设置 lora 模型的文件名，设置多个lora模型应使用不同的 lora_id.
 *
 * @param [out]
 * @param [in] param                通过Allocate申请的参数指针
 * @param [in] lora_id              lora_id(> 0)应使用较小的整数
 * @param [in] file_name            lora模型文件
 * @return
 * @note
 */
    INFERLET_API void inferlet_parameter_set_low_rank_adapter(void *param, int lora_id, const char *file_name);

    /**
 * @brief 设置加载模型接口的响应模式.
 *
 * @param [out]
 * @param [in] param                通过Allocate申请的参数指针
 * @param [in] mode                 配置为1, 为同步接口; 默认为 0, 为异步接口.
 * @return
 * @note
 */
    INFERLET_API void inferlet_parameter_set_load_from_file_sync(void *param, int mode);

    /**
 * @brief 设置模型推理运行的配置文件.
 *
 * @param [out]
 * @param [in] param                通过Allocate申请的参数指针
 * @param [in] file_name            模型推理运行配置文件(json格式).
 * @return
 * @note 图像推理配置示例：{"schema_version":1,"vision":{"mmproj":"mmproj.gguf"}}。
 *       mmproj 相对路径以 JSON 所在目录为基准，必须与加载的视觉语言 GGUF 模型配套。
 */
    INFERLET_API void inferlet_parameter_set_run_config_file(void *param, const char *file_name);

    /**
 * @brief 编码文本为tokens
 * @param [in]             handle: Inferlet句柄
 * @param [in]               data: 被编码的文本(文本本身需为UTF-8格式)
 * @param [in]        text_length: 被编码的文本长度的字节数(不包含结束符)
 * @param [out]            tokens: 输出tokens的存储地址(内存需调用者申请)
 * @param [in]   max_token_length: 输出tokens存储地址所能存储的最大的tokens个数
 * @param [out] real_token_length: 真实输出的tokens个数
                                   编码时，如果超出max_token_length，输出数据会被截断，
                                   *real_token_length 表示截断前的 token 个数.
 * @param [in]  add_special_token: 编码文本时是否添加特殊token 1:添加 0:不添加
 * @param [in]      parse_special: 1:识别文本中的特殊文本并编码为相应特殊token
                                   0:把文本全部按纯文本进行编码
 * @return 0 表示成功, <0 表示失败。
*/
    INFERLET_API int inferlet_tokenizer_encode(void *handle, const char *text, int text_length, int *tokens,
                                          int max_token_length, int *real_token_length, int add_special_token, int parse_special);

    /**
 * @brief 解码tokens为文本
 * @param [in]             handle: Inferlet句柄
 * @param [in]             tokens: 输入tokens的存储地址
 * @param [in]       token_length: 输入的tokens个数
 * @param [out]              text: 输出解码的文本存储地址(内存需调用者申请)
 * @param [in]    max_text_length: 输出解码的文本存储地址所能存储的最大字节数
 * @param [out]  real_text_length: 真实输出的文本长度的字节数(不包含结束符)
                                   解码时，如果超出 max_text_length，输出会被截断
                                   *real_text_length 表示截断前的长度.
 * @param [in] skip_special_token: 解码时是否跳过特殊token 1:跳过; 0:不跳过
 * @return 0 表示成功, <0 表示失败。
 * @note token ID 必须位于模型词表范围内；越界返回 INFERLET_E_INVALID_ARGUMENT，
 *       *real_text_length 为 0，text 缓冲区保持不变。
*/
    INFERLET_API int inferlet_tokenizer_decode(void *handle, const int *tokens, int token_length, char *text,
                                          int max_text_length, int *real_text_length, int skip_special_tokens);

#ifdef __cplusplus
}
#endif

#endif // INFERLET_H_INCLUDED

# tests/CMakeLists.txt

include_directories(
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/vendor
    ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/vendor/nlohmann
)
add_compile_options(-UNDEBUG)

# ABI Header syntax tests
set(INFERLET_C_ABI_WARNING_FLAGS)
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    list(APPEND INFERLET_C_ABI_WARNING_FLAGS -Wall -Wextra -Wpedantic -Wstrict-prototypes -Werror)
endif()
add_test(NAME abi_header_c_syntax
    COMMAND ${CMAKE_C_COMPILER} -std=c11 ${INFERLET_C_ABI_WARNING_FLAGS} -I${CMAKE_SOURCE_DIR}/include -fsyntax-only ${CMAKE_CURRENT_SOURCE_DIR}/abi/header_c.c
)
add_test(NAME abi_header_cpp_syntax
    COMMAND ${CMAKE_CXX_COMPILER} -std=c++17 -I${CMAKE_SOURCE_DIR}/include -fsyntax-only ${CMAKE_CURRENT_SOURCE_DIR}/abi/header_cpp.cpp
)

# ABI Executables linking Inferlet
add_executable(test_abi_c abi/header_c.c)
target_link_libraries(test_abi_c PRIVATE Inferlet)
add_test(NAME abi_c_link COMMAND test_abi_c)

add_executable(test_abi_cpp abi/header_cpp.cpp)
target_link_libraries(test_abi_cpp PRIVATE Inferlet)
add_test(NAME abi_cpp_link COMMAND test_abi_cpp)

# Symbol verification test
add_test(NAME abi_symbols_check
    COMMAND ${CMAKE_COMMAND}
        -DARCHIVE_FILE=$<TARGET_FILE:Inferlet>
        -DEXPECTED_SYMBOLS=${CMAKE_CURRENT_SOURCE_DIR}/abi/expected_symbols.txt
        -P ${CMAKE_CURRENT_SOURCE_DIR}/abi/test_symbols.cmake
)

# Unit tests
add_executable(test_validation unit/test_validation.cpp)
target_link_libraries(test_validation PRIVATE Inferlet)
add_test(NAME unit_validation COMMAND test_validation)

add_executable(test_runtime unit/test_runtime.cpp)
target_link_libraries(test_runtime PRIVATE Inferlet)
add_test(NAME unit_runtime COMMAND test_runtime)

add_executable(test_model_config unit/test_model_config.cpp)
target_link_libraries(test_model_config PRIVATE Inferlet)
add_test(NAME unit_model_config COMMAND test_model_config)

add_executable(test_task_input unit/test_task_input.cpp)
target_link_libraries(test_task_input PRIVATE Inferlet)
add_test(NAME unit_task_input COMMAND test_task_input)

add_executable(test_prompt_builder unit/test_prompt_builder.cpp)
target_link_libraries(test_prompt_builder PRIVATE Inferlet)
add_test(NAME unit_prompt_builder COMMAND test_prompt_builder)

add_executable(test_sampler unit/test_sampler.cpp)
target_link_libraries(test_sampler PRIVATE Inferlet)
add_test(NAME unit_sampler COMMAND test_sampler)

add_executable(test_task_state unit/test_task_state.cpp)
target_link_libraries(test_task_state PRIVATE Inferlet)
add_test(NAME unit_task_state COMMAND test_task_state)

add_executable(test_result_sink unit/test_result_sink.cpp)
target_link_libraries(test_result_sink PRIVATE Inferlet)
add_test(NAME unit_result_sink COMMAND test_result_sink)


# Integration tests
add_executable(test_tokenizer integration/test_tokenizer.cpp)
target_link_libraries(test_tokenizer PRIVATE Inferlet)
add_test(NAME integration_tokenizer COMMAND test_tokenizer)

add_executable(test_sync_generation integration/test_sync_generation.cpp)
target_link_libraries(test_sync_generation PRIVATE Inferlet)
add_test(NAME integration_sync_generation COMMAND test_sync_generation)

add_executable(test_verbose_output integration/test_verbose_output.cpp)
target_link_libraries(test_verbose_output PRIVATE Inferlet)
add_test(NAME integration_verbose_output COMMAND test_verbose_output)

add_executable(test_async integration/test_async.cpp)
target_link_libraries(test_async PRIVATE Inferlet)
add_test(NAME integration_async COMMAND test_async)

add_executable(test_async_load integration/test_async_load.cpp)
target_link_libraries(test_async_load PRIVATE Inferlet)
add_test(NAME integration_async_load COMMAND test_async_load)

add_executable(test_stream integration/test_stream.cpp)
target_link_libraries(test_stream PRIVATE Inferlet)
add_test(NAME integration_stream COMMAND test_stream)

add_executable(test_error_paths integration/test_error_paths.cpp)
target_link_libraries(test_error_paths PRIVATE Inferlet)
add_test(NAME integration_error_paths COMMAND test_error_paths)
set_tests_properties(integration_error_paths PROPERTIES TIMEOUT 300)

add_executable(test_step15 integration/test_step15.cpp)
target_link_libraries(test_step15 PRIVATE Inferlet)
add_test(NAME integration_step15 COMMAND test_step15)
set_tests_properties(integration_step15 PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(test_step16 integration/test_step16.cpp)
target_link_libraries(test_step16 PRIVATE Inferlet)
add_test(NAME integration_step16 COMMAND test_step16)

add_executable(test_multimodal integration/test_multimodal.cpp)
target_link_libraries(test_multimodal PRIVATE Inferlet)
add_test(NAME integration_multimodal COMMAND test_multimodal)

add_executable(test_models_folder integration/test_models_folder.cpp)
target_link_libraries(test_models_folder PRIVATE Inferlet)
add_test(NAME integration_models_folder COMMAND test_models_folder)
set_tests_properties(integration_models_folder PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# Consumer and packaging verification test
add_test(NAME test_consumer_package
    COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/consumer/test_consumer_package.sh ${CMAKE_BINARY_DIR}
)


# Optional real-image integration; missing fixtures are reported as skipped.
add_executable(test_vision integration/test_vision.c)
target_link_libraries(test_vision PRIVATE Inferlet)
add_test(NAME integration_vision COMMAND test_vision)
set_tests_properties(integration_vision PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)

add_executable(test_all_interfaces integration/test_all_interfaces.cpp)
target_link_libraries(test_all_interfaces PRIVATE Inferlet)
add_test(NAME integration_all_interfaces COMMAND test_all_interfaces)
set_tests_properties(integration_all_interfaces PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

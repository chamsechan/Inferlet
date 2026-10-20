# cmake/InferletOptions.cmake

# Compiler standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Force position independent code for static libraries
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Configure upstream llama.cpp options to avoid unused targets and network
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_STANDALONE OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_MTMD ON CACHE BOOL "" FORCE)
set(MTMD_VIDEO OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_UI OFF CACHE BOOL "" FORCE)
set(LLAMA_USE_PREBUILT_UI OFF CACHE BOOL "" FORCE)
set(LLAMA_TOOLS_INSTALL OFF CACHE BOOL "" FORCE)
set(LLAMA_TESTS_INSTALL OFF CACHE BOOL "" FORCE)
set(LLAMA_ALL_WARNINGS OFF CACHE BOOL "" FORCE)
set(GGML_ALL_WARNINGS OFF CACHE BOOL "" FORCE)
set(GGML_METAL OFF CACHE BOOL "" FORCE)
# Distributed static archives must not target the build host CPU; with native
# off, ggml uses its portable x86-64 baseline (AVX2/FMA/F16C, no AVX-512).
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)

# Inferlet build options
option(INFERLET_BUILD_TESTS "Build Inferlet tests" OFF)
option(INFERLET_ENABLE_ASAN "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(INFERLET_ENABLE_TSAN "Enable ThreadSanitizer" OFF)

if (INFERLET_ENABLE_ASAN AND INFERLET_ENABLE_TSAN)
    message(FATAL_ERROR "Cannot enable both ASan and TSan at the same time")
endif()

# Reset upstream cache entries when instrumentation is disabled or switched,
# so exported link requirements match the actual archive contents.
set(LLAMA_SANITIZE_ADDRESS ${INFERLET_ENABLE_ASAN} CACHE BOOL "" FORCE)
set(LLAMA_SANITIZE_UNDEFINED ${INFERLET_ENABLE_ASAN} CACHE BOOL "" FORCE)
set(LLAMA_SANITIZE_THREAD ${INFERLET_ENABLE_TSAN} CACHE BOOL "" FORCE)

if (INFERLET_ENABLE_ASAN)
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -g -O1)
    add_link_options(-fsanitize=address,undefined)
endif()

if (INFERLET_ENABLE_TSAN)
    add_compile_options(-fsanitize=thread -g)
    add_link_options(-fsanitize=thread)
endif()

# Project warning flags for Inferlet code
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(INFERLET_CXX_WARNINGS -Wall -Wextra -Wpedantic -Wno-unused-parameter)
else()
    set(INFERLET_CXX_WARNINGS "")
endif()

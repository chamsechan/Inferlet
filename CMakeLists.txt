cmake_minimum_required(VERSION 3.20)
project(Inferlet VERSION 0.1.0 LANGUAGES C CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(cmake/InferletOptions.cmake)
include(cmake/InferletFatArchive.cmake)

# Fixed llama.cpp commit tracking
set(INFERLET_PINNED_LLAMA_COMMIT "8b4b3558f1459c13e4aa38d5c94d306a00dc6acd")
message(STATUS "Inferlet pinned llama.cpp commit: ${INFERLET_PINNED_LLAMA_COMMIT}")

# Add third-party llama.cpp
add_subdirectory(third_party/llama.cpp EXCLUDE_FROM_ALL)

find_package(Threads REQUIRED)
if(GGML_OPENMP)
    find_package(OpenMP)
endif()

# Inferlet core library sources
set(INFERLET_SOURCES
    src/api/inferlet_c_api.cpp
    src/core/inference_task.cpp
    src/core/model_config.cpp
    src/core/model_registry.cpp
    src/core/runtime.cpp
    src/core/scheduler.cpp
    src/core/stream_sink.cpp
    src/core/task_input.cpp
    src/engine/beam_search.cpp
    src/engine/generator.cpp
    src/engine/guided_choice.cpp
    src/engine/llama_engine.cpp
    src/engine/lora_registry.cpp
    src/engine/model_prompt_adapter.cpp
    src/engine/prompt_builder.cpp
    src/engine/sampler.cpp
    src/engine/token_codec.cpp
    src/multimodal/multimodal_backend.cpp
    src/multimodal/mtmd_backend.cpp
    src/multimodal/visual_cache.cpp
    src/support/json_config.cpp
    src/support/last_error.cpp
    src/support/logging.cpp
    src/support/object_registry.cpp
    src/support/validation.cpp
)

add_library(Inferlet STATIC ${INFERLET_SOURCES})
add_library(Inferlet::Inferlet ALIAS Inferlet)
set_target_properties(Inferlet PROPERTIES OUTPUT_NAME inferlet)

target_include_directories(Inferlet
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/vendor
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/vendor/nlohmann
)

target_compile_options(Inferlet PRIVATE ${INFERLET_CXX_WARNINGS})
target_link_libraries(Inferlet PRIVATE
    $<BUILD_INTERFACE:llama>
    $<BUILD_INTERFACE:mtmd>
    $<BUILD_INTERFACE:Threads::Threads>
    $<BUILD_INTERFACE:${CMAKE_DL_LIBS}>
    $<BUILD_INTERFACE:m>
)

if (GGML_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(Inferlet PRIVATE $<BUILD_INTERFACE:OpenMP::OpenMP_CXX>)
    set(INFERLET_HAS_OPENMP TRUE)
    set(PC_OPENMP_LIBS "-fopenmp")
else()
    set(INFERLET_HAS_OPENMP FALSE)
    set(PC_OPENMP_LIBS "")
endif()

# Bundle dependencies into a single fat archive
inferlet_add_fat_archive(Inferlet mtmd vendor-hash llama ggml ggml-base ggml-cpu)

# Sanitizer flags for pkg-config and CMake exported config
if (INFERLET_ENABLE_ASAN)
    set(PC_SANITIZER_CFLAGS "-fsanitize=address,undefined")
    set(PC_SANITIZER_LDFLAGS "-fsanitize=address,undefined")
elseif(INFERLET_ENABLE_TSAN)
    set(PC_SANITIZER_CFLAGS "-fsanitize=thread")
    set(PC_SANITIZER_LDFLAGS "-fsanitize=thread")
else()
    set(PC_SANITIZER_CFLAGS "")
    set(PC_SANITIZER_LDFLAGS "")
endif()

# Configure pkg-config
file(RELATIVE_PATH INFERLET_PC_PREFIX_RELATIVE
    "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/pkgconfig"
    "${CMAKE_INSTALL_PREFIX}")
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/inferlet.pc.in
    ${CMAKE_CURRENT_BINARY_DIR}/inferlet.pc
    @ONLY
)

# Configure CMake Package Config
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/InferletConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/InferletConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Inferlet
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/InferletConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

# Installation targets
install(TARGETS Inferlet
    EXPORT InferletTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Install static library into architecture-specific subdirectory under lib/ (x64, arm, mac)
if(DEFINED INFERLET_ARCH_DIR)
    set(INFERLET_TARGET_ARCH_SUBDIR "${INFERLET_ARCH_DIR}")
elseif(APPLE OR CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(INFERLET_TARGET_ARCH_SUBDIR "mac")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(INFERLET_TARGET_ARCH_SUBDIR "arm")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64")
    set(INFERLET_TARGET_ARCH_SUBDIR "x64")
else()
    set(INFERLET_TARGET_ARCH_SUBDIR "${CMAKE_SYSTEM_PROCESSOR}")
endif()

install(FILES $<TARGET_FILE:Inferlet>
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/${INFERLET_TARGET_ARCH_SUBDIR}
)

install(FILES include/inferlet.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT InferletTargets
    FILE InferletTargets.cmake
    NAMESPACE Inferlet::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Inferlet
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/InferletConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/InferletConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Inferlet
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/inferlet.pc
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
)

install(FILES third_party/llama.cpp/LICENSE
    DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/Inferlet
    RENAME LICENSE.llama.cpp
)

install(FILES doc/third_party.md
    DESTINATION ${CMAKE_INSTALL_DATADIR}/doc/Inferlet
)
install(FILES doc/vision.md
    DESTINATION ${CMAKE_INSTALL_DATADIR}/doc/Inferlet
)
install(FILES doc/mtmd_licenses.txt
    DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/Inferlet
)

# Optional examples
option(INFERLET_BUILD_EXAMPLES "Build Inferlet examples" OFF)
if (INFERLET_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Tests
if (INFERLET_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

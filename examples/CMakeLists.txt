cmake_minimum_required(VERSION 3.20)
project(inferlet_examples CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(TARGET Inferlet)
    set(INFERLET_LIB Inferlet)
else()
    find_package(Inferlet REQUIRED)
    set(INFERLET_LIB Inferlet::Inferlet)
endif()

add_executable(example_sync sync.cpp)
target_link_libraries(example_sync PRIVATE ${INFERLET_LIB})

add_executable(example_async async.cpp)
target_link_libraries(example_async PRIVATE ${INFERLET_LIB})

add_executable(example_stream stream.cpp)
target_link_libraries(example_stream PRIVATE ${INFERLET_LIB})

add_executable(example_tokenizer tokenizer.cpp)
target_link_libraries(example_tokenizer PRIVATE ${INFERLET_LIB})

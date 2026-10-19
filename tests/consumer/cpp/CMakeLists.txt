cmake_minimum_required(VERSION 3.20)
project(consumer_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Inferlet REQUIRED)

add_executable(consumer_cpp main.cpp)
target_compile_options(consumer_cpp PRIVATE -UNDEBUG)
target_link_libraries(consumer_cpp PRIVATE Inferlet::Inferlet)

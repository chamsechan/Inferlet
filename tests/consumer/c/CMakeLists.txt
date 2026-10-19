cmake_minimum_required(VERSION 3.20)
project(consumer_c C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

find_package(Inferlet REQUIRED)

add_executable(consumer_c main.c)
target_compile_options(consumer_c PRIVATE -UNDEBUG)
target_link_libraries(consumer_c PRIVATE Inferlet::Inferlet)

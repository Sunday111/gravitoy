cmake_minimum_required(VERSION 3.20)
include(set_compiler_options)
set(module_source_files
    ${CMAKE_CURRENT_SOURCE_DIR}/code/private/main.cpp)
add_executable(gravitoy ${module_source_files})
set_generic_compiler_options(gravitoy PRIVATE)
target_link_libraries(gravitoy PUBLIC gravitoy_lib)
target_include_directories(gravitoy PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/code/public)
target_include_directories(gravitoy PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/code/private)

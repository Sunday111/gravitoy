cmake_minimum_required(VERSION 3.20)
include(set_compiler_options)
set(module_source_files
    ${CMAKE_CURRENT_SOURCE_DIR}/code/private/app.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/code/public/gravitoy/app.hpp)
add_library(gravitoy_lib STATIC ${module_source_files})
set_generic_compiler_options(gravitoy_lib PRIVATE)
target_link_libraries(gravitoy_lib PUBLIC klgl)
target_include_directories(gravitoy_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/code/public)
target_include_directories(gravitoy_lib PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/code/private)
add_custom_target(gravitoy_lib_copy_files ALL
    ${CMAKE_COMMAND} -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/content" ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/content
)
add_dependencies(gravitoy_lib_copy_files gravitoy_lib)

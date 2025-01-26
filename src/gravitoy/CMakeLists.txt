cmake_minimum_required(VERSION 3.20)
include(set_compiler_options)
set(module_source_files
    ${CMAKE_CURRENT_SOURCE_DIR}/code/private/gravitoy.cpp)
add_executable(gravitoy ${module_source_files})
set_generic_compiler_options(gravitoy PRIVATE)
target_link_libraries(gravitoy PUBLIC klgl)
target_include_directories(gravitoy PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/code/public)
target_include_directories(gravitoy PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/code/private)
add_custom_target(gravitoy_copy_files ALL
    ${CMAKE_COMMAND} -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/content" ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/content
)
add_dependencies(gravitoy_copy_files gravitoy)

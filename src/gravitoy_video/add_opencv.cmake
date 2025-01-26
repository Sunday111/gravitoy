find_package(OpenCV REQUIRED core videoio highgui)
target_link_libraries(gravitoy_video PRIVATE ${OpenCV_LIBS})

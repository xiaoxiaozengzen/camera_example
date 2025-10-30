cmake_minimum_required(VERSION 3.23)
project(image_example)

find_package(OpenCV REQUIRED)
find_package(libyuv REQUIRED)

add_executable(jpeg_example jpeg_example.cpp)

add_executable(yuv_example yuv_example.cpp)
target_include_directories(yuv_example PRIVATE ${OpenCV_INCLUDE_DIRS} ${libyuv_INCLUDE_DIRS})
target_link_libraries(yuv_example ${OpenCV_LIBRARIES} ${libyuv_LIBRARIES})
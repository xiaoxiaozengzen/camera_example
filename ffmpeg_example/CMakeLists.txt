cmake_minimum_required(VERSION 3.23)
project(ffmpeg_example)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ffmpeg REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(ffmpeg_codec_example ffmpeg_codec_example.cpp)
target_include_directories(ffmpeg_codec_example PRIVATE ${ffmpeg_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

message(STATUS "ffmpeg_LIBRARIES : ${ffmpeg_LIBRARIES}")
target_link_libraries(ffmpeg_codec_example ${ffmpeg_LIBRARIES} ${OpenCV_LIBRARIES})


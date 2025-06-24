cmake_minimum_required(VERSION 3.23)
project(camera_example)

file(GLOB SOURCES "src/*.cpp")
add_executable(camera_example ${SOURCES})

add_subdirectory(jpeg)
add_subdirectory(ffmpeg_example)
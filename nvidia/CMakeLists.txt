project(nvidia_camera_example)

find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDAToolkit_FOUND: ${CUDAToolkit_FOUND}")
message(STATUS "CUDAToolkit_DIR : ${CUDAToolkit_DIR}")
message(STATUS "CUDAToolkit version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDAToolkit include dirs: ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "CUDAToolkit libraries: ${CUDAToolkit_LIBRARIES}")
message(STATUS "CUDAToolkit library dir: ${CUDAToolkit_LIBRARY_DIR}")

message(STATUS "CUDAToolkit_ROOT: ${CUDAToolkit_ROOT}")
message(STATUS "ENV CUDAToolkit_ROOT: $ENV{CUDAToolkit_ROOT}")
message(STATUS "CUDAToolkit_ROOT_DIR: ${CUDAToolkit_ROOT_DIR}")
message(STATUS "CUDAToolkit_NVCC_EXECUTABLE: ${CUDAToolkit_NVCC_EXECUTABLE}")

message(STATUS "CMAKE_SYSROOT_LINK : ${CMAKE_SYSROOT_LINK}")
message(STATUS "CMAKE_SYSROOT     : ${CMAKE_SYSROOT}")


add_executable(nv_image_resize nv_image_resize.cpp)
target_include_directories(nv_image_resize PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_directories(nv_image_resize PUBLIC $ENV{CUDAToolkit_ROOT}/lib64)
target_link_libraries(nv_image_resize PUBLIC cudart nppial nppicc nppidei nppif nppig nppim nvjpeg)
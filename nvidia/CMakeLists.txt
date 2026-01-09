project(nvidia_camera_example CXX CUDA)

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

find_package(ffmpeg REQUIRED)

add_executable(nvjpeg_image_resize "resize/nvjpeg_image_resize.cpp")
target_include_directories(nvjpeg_image_resize PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_directories(nvjpeg_image_resize PUBLIC $ENV{CUDAToolkit_ROOT}/lib64)
target_link_libraries(nvjpeg_image_resize PUBLIC cudart nppial nppicc nppidei nppif nppig nppim nvjpeg)

set(CUDA_STREAM_EVENT "cudastream_event/cuda_stream_event.cu")
add_executable(cuda_stream_event ${CUDA_STREAM_EVENT})
target_include_directories(cuda_stream_event PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_directories(cuda_stream_event PUBLIC $ENV{CUDAToolkit_ROOT}/lib64)
target_link_libraries(cuda_stream_event PUBLIC CUDA::cudart)

set(NVCODEV_DIR "/mnt/workspace/cgz_workspace/ThreeParty/Video_Codec_SDK_13.0.19")
file(GLOB NV_DECODE_SRCS "nv_decode/*.cpp")
add_executable(nv_video_decode ${NV_DECODE_SRCS})

target_include_directories(
    nv_video_decode PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
    ${NVCODEV_DIR}/Interface
    ${CMAKE_CURRENT_SOURCE_DIR}/nv_decode
    ${ffmpeg_INCLUDE_DIRS}
)

target_link_directories(
    nv_video_decode PUBLIC 
    ${NVCODEV_DIR}/Lib/linux/stubs/x86_64
    $ENV{CUDAToolkit_ROOT}/lib64
)
target_link_libraries(
    nv_video_decode PUBLIC
    CUDA::cudart
    CUDA::cufft
    CUDA::cuda_driver
    nvcuvid 
    nvidia-encode
    ${ffmpeg_LIBRARIES}
)
get_target_property(cuda_driver_IID CUDA::cuda_driver INTERFACE_INCLUDE_DIRECTORIES)
message(STATUS "cuda_driver_IID: ${cuda_driver_IID}")
get_target_property(cuda_driver_NAME CUDA::cuda_driver NAME)
message(STATUS "cuda_driver_NAME: ${cuda_driver_NAME}")
get_target_property(cuda_driver_location CUDA::cuda_driver IMPORTED_LOCATION)
message(STATUS "cuda_driver_location: ${cuda_driver_location}")
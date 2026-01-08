#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>
#include <cuda.h>

/**
 * @brief 
 * enum cudaMemcpyKind {
 *     cudaMemcpyHostToHost = 0,    // 主机到主机的内存拷贝
 *     cudaMemcpyHostToDevice = 1,  // 主机到设备的内存拷贝
 *     cudaMemcpyDeviceToHost = 2,  // 设备到主机的内存拷贝
 *     cudaMemcpyDeviceToDevice = 3,  // 设备到设备的内存拷贝
 *     cudaMemcpyDefault = 4        // 默认内存拷贝方式，由CUDA运行时决定
 * };
 */

/**
 * @brief 
 * enum cudaError {
 *     cudaSuccess = 0,                     // 操作成功
 *     cudaErrorInvalidValue = 1,           // 无效的值
 *     cudaErrorMemoryAllocation = 2,       // 内存分配失败
 *     cudaErrorInitializationError = 3,    // 初始化错误
 *     ...                               // 其他错误代码
 * }; 
 * typedef __device_builtin__ enum cudaError cudaError_t;
 */

/**
 * @brief 
 * typedef __device_builtin__ struct CUstream_st *cudaStream_t
 * @note CUDA 流（cudaStream_t）是一种用于管理和调度 GPU 上异步操作的机制。
 *       它允许多个操作（如内存拷贝和内核执行）在同一时间内并行进行，从而提高 GPU 的利用率和整体性能。
 *       一个stream对应一个执行队列
 *
 * typedef __device_builtin__ struct CUevent_st *cudaEvent_t;
 */

#define CHECK_CUDA(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                           \
        if (err != cudaSuccess) {                                         \
            std::cerr << "CUDA error in " << __FILE__ << " at line "      \
                      << __LINE__ << ": " << cudaGetErrorString(err)      \
                      << " (" << err << ")" << std::endl;                 \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

int main(int argc, char** argv) {
    std::string image_input_yuv = "/mnt/workspace/cgz_workspace/Exercise/camera_example/input/650_yuv420p.yuv";
    int width = 1080;
    int height = 1920;

    std::ifstream infile(image_input_yuv, std::ios::binary | std::ios::ate);
    if(!infile.is_open()) {
        std::cerr << "Failed to open input file: " << image_input_yuv << std::endl;
        return -1;
    }

    std::size_t data_size = infile.tellg();
    std::cout << "data_size: " << data_size
              << ", width * height * 3 / 2: " << width * height * 3 / 2
              << std::endl;
    infile.seekg(0, std::ios::end);
    std::vector<char> buffer(data_size);
    infile.read(buffer.data(), data_size);
    infile.close();

    uint8_t* h_src = nullptr;
    uint8_t* h_dst = nullptr;
    CHECK_CUDA(cudaMallocHost(reinterpret_cast<void**>(&h_src), data_size));
    CHECK_CUDA(cudaMallocHost(reinterpret_cast<void**>(&h_dst), data_size));

    uint8_t* d_buf = nullptr;
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_buf), data_size));

    // 创建 CUDA 流
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaFreeHost(h_src));
    CHECK_CUDA(cudaFreeHost(h_dst));
    CHECK_CUDA(cudaFree(d_buf));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaStreamDestroy(stream));

    std::cout << "Success!" << std::endl;
}
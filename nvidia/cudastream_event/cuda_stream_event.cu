#include <iostream>
#include <vector>
#include <fstream>

#include <cuda_runtime.h>
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
 * typedef struct cudaArray *cudaArray_t;
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

/************************************ 1.cudaStream cudaEvent ************************************/

// kernel函数定义
__global__ void kernel1(int64_t *data, int64_t repeat) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    for (size_t i = 0; i < repeat; i++)
    {
        data[idx] += 1;
    }
}

__global__ void kernel2(int64_t *data, int64_t repeat) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    for (size_t i = 0; i < repeat; i++)
    {
        data[idx] += 2;
    }
}

__global__ void kernel3(int64_t *data, int64_t repeat) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    for (size_t i = 0; i < repeat; i++)
    {
        data[idx] -= 1;
    }
}

void cuda_stream_event() {
    const int dataSize = 1024;
    const int printSize = 10;
    int64_t *h_data = new int64_t[dataSize];
    int64_t *d_data1, *d_data2;

    for (int i = 0; i < dataSize; i++) {
        h_data[i] = 0;
    }

    cudaMalloc((void**)&d_data1, dataSize * sizeof(int64_t));
    cudaMalloc((void**)&d_data2, dataSize * sizeof(int64_t));

    // 将数据从主机传输到设备
    cudaMemcpy(d_data1, h_data, dataSize * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data2, h_data, dataSize * sizeof(int64_t), cudaMemcpyHostToDevice);

    // 定义网格和块的维度
    dim3 blockDim(256);
    dim3 gridDim((dataSize + blockDim.x - 1) / blockDim.x);

    // 创建流和事件
    cudaStream_t stream1, stream2;
    cudaEvent_t event1, event1_stop;
    int priorityHigh, priorityLow;
    cudaDeviceGetStreamPriorityRange(&priorityLow, &priorityHigh);
    std::cout << "Stream priority range: Low=" << priorityLow << ", High=" << priorityHigh << std::endl;
    cudaStreamCreate(&stream1);
    cudaStreamCreateWithPriority(&stream2, cudaStreamDefault, priorityHigh);
    cudaEventCreate(&event1);
    cudaEventCreate(&event1_stop);

    unsigned int cudastream_flags = 0;
    /**
     * cudaStreamDefault：0，默认流标志，表示标准的流行为。
     * cudaStreamNonBlocking：1，非阻塞流标志，表示流中的操作跟默认流（stream 0）的并行的
     */
    CHECK_CUDA(cudaStreamGetFlags(stream1, &cudastream_flags));
    std::cout << "Stream1 flags: " << cudastream_flags << std::endl;

    // 定义重复次数
    const int64_t repeat = 1000;

    // 1. 指定在stream1中执行kernel1
    kernel1<<<gridDim, blockDim, 0, stream1>>>(d_data1, repeat);
    // 应该是创建一个临时kernel(对应event1)，记录这个kernel执行开始时间
    cudaEventRecord(event1, stream1);

    // 2. 在stream2中执行kernel2，等待event1完成才能执行
    cudaStreamWaitEvent(stream2, event1, 0);
    kernel2<<<gridDim, blockDim, 0, stream2>>>(d_data1, repeat);

    // 3. 在stream1中执行kernel3
    kernel3<<<gridDim, blockDim, 0, stream1>>>(d_data2, repeat);
    // 在当前host线程中阻塞，直到提交到stream中的所有事件(包含kernel操作，内存拷贝)完成后才继续往下执行
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    // 记录event1_stop对应的时间点
    cudaEventRecord(event1_stop, stream1);
    // 在当前host线程中阻塞，直到GPU到达事件event1_stop才继续往下执行
    cudaEventSynchronize(event1_stop);
    float cost_ms = 0.0f;
    cudaEventElapsedTime(&cost_ms, event1, event1_stop);
    std::cout << "Elapsed time between event1 and event1_stop: " << cost_ms << " ms" << std::endl;

    // 将数据从设备传输回主机
    cudaMemcpy(h_data, d_data1, dataSize * sizeof(int64_t), cudaMemcpyDeviceToHost);
    // 显示结果
    std::cout << "Data after kernel1 and kernel2:" << std::endl;
    for (int i = 0; i < printSize; i++) {
        std::cout << h_data[i] << " ";
    }
    std::cout << std::endl;

    // 将数据从设备传输回主机
    cudaMemcpy(h_data, d_data2, dataSize * sizeof(int64_t), cudaMemcpyDeviceToHost);
    // 显示结果
    std::cout << "Data after kernel3:" << std::endl;
    for (int i = 0; i < printSize; i++) {
        std::cout << h_data[i] << " ";
    }
    std::cout << std::endl;

    // 释放资源
    cudaFree(d_data1);
    cudaFree(d_data2);
    delete[] h_data;
    cudaEventDestroy(event1);
    cudaEventDestroy(event1_stop);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
}

int main(int argc, char* argv[]) {
    std::cout << "===================== cuda_stream_event =====================" << std::endl;
    cuda_stream_event();

    return 0;
}
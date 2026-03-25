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

/************************************ 1.cudaMemcpyAsync ************************************/

// kernel函数：将数据进行反转
__global__ void invert_kernel(uint8_t* data, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = 255u - data[idx];
}

void cuda_memcpy_async() {
    const int width = 1280;
    const int height = 720;
    const int channels = 3; // RGB
    const size_t img_bytes = static_cast<size_t>(width) * height * channels;

    /**
     * malloc：申请的是pageable内存，可能会被系统重新映射到其他物理页，导致GPU无法直接访问
     * cudaHostAlloc：申请的是pinned内存（page-locked memory），不会被系统重新映射，GPU可以直接访问，适合用于异步数据传输
     *
     * cudaHostAlloc与设备做主机<->设备的异步拷贝时，能够显著提升数据传输效率，减少延迟。
     * malloc申请的pageable内存，会被临时拷贝到一个中间的pinned内存区域，再由该区域传输到设备，增加了额外的拷贝开销。
     *
     * cudaHostAlloc的标志：
     * cudaHostAllocDefault：默认标志，表示分配的内存是页锁定的，可以用于异步数据传输
     * cudaHostAllocMapped：映射标志，表示分配的内存可以映射到设备地址空间，允许设备直接访问该内存
     *                      用于零拷贝访问（zero-copy access），适合小数据量低频访问场景
     * cudaHostAllocPortable：可移植标志，表示分配的内存可以在多个CUDA上下文中使用
     *
     * cudaMalloc跟cudaMallocAsync的区别：
     * 1. 同步与异步：
     *    cudaMalloc是一个同步函数，调用时会阻塞CPU线程直到内存分配完成；
     *    而cudaMallocAsync是一个异步函数，调用时不会阻塞CPU线程，可以与其他操作并行进行
     * 2. 内存分配方式：
     *    cudaMalloc直接在设备上分配内存，
     *    cudaMallocAsync使用CUDA memory pool进行内存管理，分配的内存会回到内存池中，而不是立即归还给操作系统
     * 3.流有序性：
     *    cudaMalloc是全局同步的，所有流都会等待cudaMalloc完成后才能继续执行；
     *    cudaMallocAsync是流有序的，只有提交cudaMallocAsync的流会等待内存分配完成，其他流可以继续执行不受影响
     */
    uint8_t* h_src = nullptr;
    uint8_t* h_dst = nullptr;
    uint8_t* data_cpu = nullptr;
    uint8_t* h_data_async = nullptr;
    CHECK_CUDA(cudaHostAlloc(&h_src, img_bytes, cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_dst, img_bytes, cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&data_cpu, img_bytes, cudaHostAllocMapped));
    CHECK_CUDA(cudaHostAlloc(&h_data_async, img_bytes, cudaHostAllocDefault));
    memset(h_dst, 0, img_bytes);
    memset(h_src, 0, img_bytes);
    memset(data_cpu, 0, img_bytes);
    memset(h_data_async, 0, img_bytes);

    // 填充数据
    for (size_t i = 0; i < img_bytes; ++i) h_src[i] = static_cast<uint8_t>(i & 0xFF);
    for (size_t i = 0; i < img_bytes; ++i) data_cpu[i] = static_cast<uint8_t>(i & 0xFF);
    for (size_t i = 0; i < img_bytes; ++i) h_data_async[i] = static_cast<uint8_t>(i & 0xFF);

    // 分配设备内存
    uint8_t* d_buf = nullptr;
    CHECK_CUDA(cudaMalloc(&d_buf, img_bytes));
    CHECK_CUDA(cudaMemset(d_buf, 0, img_bytes));

    uint8_t* data_gpu = nullptr;
    /**
     * @brief 将cudaHostAlloc申请的pinned内存，映射到设备地址空间，获取对应的设备指针
     */
    CHECK_CUDA(cudaHostGetDevicePointer(&data_gpu, data_cpu, 0));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 记录起始时间
    CHECK_CUDA(cudaEventRecord(start, stream));

    uint8_t* d_data_async = nullptr;
    /**
     * @brief 在指定的stream上异步分配设备内存
     * @param devPtr 输出参数，返回分配的设备内存指针
     * @param size   输入参数，要分配的内存大小（字节）
     * @param stream  输入参数，要执行内存分配的stream
     */
    CHECK_CUDA(cudaMallocAsync(&d_data_async, img_bytes, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_data_async, h_data_async, img_bytes, cudaMemcpyHostToDevice, stream));

     /**
     * @brief 在指定的stream上异步释放设备内存
     * @param devPtr 输入参数，要释放的设备内存指针
     * @param stream  输入参数，要执行内存释放的stream
     */

    /**
     * 相对host这是一个异步操作，cudaMemcpyAsync会立即返回，不会阻塞CPU线程
     * 但是数据传输会在指定的stream中进行，可能会与其他操作并行执行
     *
     * 要求：host上的内存必须是pinned memory（page-locked memory），否则行为未定义
     */
    CHECK_CUDA(cudaMemcpyAsync(d_buf, h_src, img_bytes, cudaMemcpyHostToDevice, stream));

    // 对于同一个stream，操作是按顺序执行的，即会保证内存拷贝完成后才执行kernel
    const int threads = 256;
    const int blocks = static_cast<int>((img_bytes + threads - 1) / threads);
    invert_kernel<<<blocks, threads, 0, stream>>>(d_buf, img_bytes);
    invert_kernel<<<blocks, threads, 0, stream>>>(data_gpu, img_bytes);
    invert_kernel<<<blocks, threads, 0, stream>>>(d_data_async, img_bytes);
    
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaMemcpyAsync(h_dst, d_buf, img_bytes, cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaMemcpyAsync(h_data_async, d_data_async, img_bytes, cudaMemcpyHostToHost, stream));

    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // 计算时间差
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    std::cout << "Roundtrip (H2D + kernel + D2H) elapsed: " << ms << " ms\n";

    // 快速验证（前几个像素）
    bool ok = true;
    for (int i = 0; i < 16; ++i) {
        std::cout << "h_src[" << i << "] = " << static_cast<int>(h_src[i])
                  << ", h_dst[" << i << "] = " << static_cast<int>(h_dst[i]) << "\n";
        if (h_dst[i] != static_cast<uint8_t>(255u - h_src[i])) {
            ok = false;
            break;
        }
    }
    std::cout << "Validation: " << (ok ? "PASS" : "FAIL") << "\n";
    for (int i = 0; i < 16; ++i) {
        std::cout << "data_cpu[" << i << "] = " << static_cast<int>(data_cpu[i]) << "\n";
    }
    for(int i = 0; i < 16; ++i) {
        std::cout << "h_data_async[" << i << "] = " << static_cast<int>(h_data_async[i]) << "\n";
    }

    // 释放资源
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_buf));
    CHECK_CUDA(cudaFreeHost(h_src));
    CHECK_CUDA(cudaFreeHost(h_dst));
    CHECK_CUDA(cudaFreeHost(data_cpu));

    /**
     * @brief 在指定的stream上异步释放设备内存
     * @param devPtr 输入参数，要释放的设备内存指针
     * @param stream  输入参数，要执行内存释放的stream
     */
    CHECK_CUDA(cudaFreeAsync(d_data_async, stream));
    CHECK_CUDA(cudaStreamDestroy(stream));
    data_gpu = nullptr;
}

int main(int argc, char* argv[]) {
    std::cout << "===================== cuda_memcpy_async =====================" << std::endl;
    cuda_memcpy_async();

    return 0;
}
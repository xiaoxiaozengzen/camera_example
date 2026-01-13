#include <iostream>
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
     * maclloc申请的pageable内存，会被临时拷贝到一个中间的pinned内存区域，再由该区域传输到设备，增加了额外的拷贝开销。
     */
    uint8_t* h_src = nullptr;
    uint8_t* h_dst = nullptr;
    CHECK_CUDA(cudaHostAlloc(&h_src, img_bytes, cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_dst, img_bytes, cudaHostAllocDefault));
    memset(h_dst, 0, img_bytes);
    memset(h_src, 0, img_bytes);

    // 填充数据
    for (size_t i = 0; i < img_bytes; ++i) h_src[i] = static_cast<uint8_t>(i & 0xFF);

    uint8_t* d_buf = nullptr;
    CHECK_CUDA(cudaMalloc(&d_buf, img_bytes));
    CHECK_CUDA(cudaMemset(d_buf, 0, img_bytes));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 记录起始时间
    CHECK_CUDA(cudaEventRecord(start, stream));

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
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaMemcpyAsync(h_dst, d_buf, img_bytes, cudaMemcpyDeviceToHost, stream));

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

    // 释放资源
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(d_buf));
    CHECK_CUDA(cudaFreeHost(h_src));
    CHECK_CUDA(cudaFreeHost(h_dst));
}

/************************************ 3.cudaArray ************************************/
/**
 * struct __builtin_align__(16) float4 {
 *     float x;
 *     float y;
 *     float z;
 *     float w;
 * };
 */
/**
 * strut cudaChannelFormatDesc {
 *     int x; // 第一个分量的位数
 *     int y; // 第二个分量的位数
 *     int z; // 第三个分量的位数
 *     int w; // 第四个分量的位数
 *     enum cudaChannelFormatKind f; // 分量的类型（如浮点型、整数型等）
 * };
 * enum cudaChannelFormatKind {
 *     cudaChannelFormatKindSigned = 0,    // 有符号整数
 *     cudaChannelFormatKindUnsigned = 1,  // 无符号整数
 *     cudaChannelFormatKindFloat = 2,     // 浮点数
 *     cudaChannelFormatKindNone = 3       // 无类型
 *     cudaChannelFormatKindNV12 = 4    // unsigned 8-bit integer ,plannar 4:2:0 YUV
 *     cudaChannelFormatKindUnsignedNormalized8X1 = 5 // 1 channel, 8-bit unsigned normalized
 *     ... // 其他格式
 * };
 */
__global__ void sampleKernel(cudaTextureObject_t texObj, float4* out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // Use unnormalized coordinates (texDesc.normalizedCoords = 0)
    float u = x + 0.5f;
    float v = y + 0.5f;
    float4 val = tex2D<float4>(texObj, u, v);
    out[y * width + x] = val;
}

void cudaArray_example() {
    const int width = 512;
    const int height = 256;
    const size_t numPixels = (size_t)width * height;

    // 在host上分配并初始化图像数据，按照RGBA格式存储
    float4* h_img = (float4*)malloc(numPixels * sizeof(float4));
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float u = float(x) / (width - 1);
            float v = float(y) / (height - 1);
            h_img[y * width + x] = make_float4(u, v, 0.0f, 1.0f);
        }
    }

    // 创建通道格式描述符，用于定义cudaArray中数据的格式
    // 这里使用float4类型，则每个分量占32位，分量类型是float
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
    cudaArray_t cuArray;
    CHECK_CUDA(cudaMallocArray(&cuArray, &channelDesc, width, height));

    // 从host拷贝数据到cudaArray
    CHECK_CUDA(cudaMemcpy2DToArray(
        cuArray,              // dst
        0, 0,                 // dstX, dstY
        h_img,                // src
        width * sizeof(float4), // srcPitch
        width * sizeof(float4), // width in bytes
        height,               // height
        cudaMemcpyHostToDevice));

    // Create resource descriptor
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    // Create texture descriptor
    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear; // or cudaFilterModePoint
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0; // use integer coordinates

    // Create texture object
    cudaTextureObject_t texObj = 0;
    CHECK_CUDA(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr));

    // Device output buffer
    float4* d_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_out, numPixels * sizeof(float4)));
    // Launch kernel to sample texture into d_out
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    sampleKernel<<<grid, block>>>(texObj, d_out, width, height);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy back and verify a few pixels
    float4* h_out = (float4*)malloc(numPixels * sizeof(float4));
    CHECK_CUDA(cudaMemcpy(h_out, d_out, numPixels * sizeof(float4), cudaMemcpyDeviceToHost));

    printf("Sample checks:\n");
    for (int i = 0; i < 5; ++i) {
        int x = i * (width / 5);
        int y = height / 2;
        float4 a = h_img[y * width + x];
        float4 b = h_out[y * width + x];
        printf(" (%d,%d) host=(%f,%f,%f,%f) gpu=(%f,%f,%f,%f)\n",
               x, y, a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w);
    }

    // Cleanup
    CHECK_CUDA(cudaDestroyTextureObject(texObj));
    CHECK_CUDA(cudaFreeArray(cuArray));
    CHECK_CUDA(cudaFree(d_out));
    free(h_img);
    free(h_out);
}

int main(int argc, char* argv[]) {
    std::cout << "===================== cuda_stream_event =====================" << std::endl;
    cuda_stream_event();
    std::cout << "===================== cuda_memcpy_async =====================" << std::endl;
    cuda_memcpy_async();
    std::cout << "===================== cudaArray_example =====================" << std::endl;
    cudaArray_example();
    return 0;
}
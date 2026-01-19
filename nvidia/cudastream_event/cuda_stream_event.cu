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

/************************************ 3. Crop YUV Image ************************************/
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
__global__ void cropYUV_kernel(cudaTextureObject_t texY, cudaTextureObject_t texU, cudaTextureObject_t texV,
                               unsigned char* outY, unsigned char* outU, unsigned char* outV,
                               int in_w, int in_h, int crop_x, int crop_y, int crop_w, int crop_h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= crop_w || y >= crop_h) return;

    // Y plane: 1:1
    int src_x = crop_x + x;
    int src_y = crop_y + y;
    /**
     * @brief 在device上从纹理对象堆区一个texel值
     * @note 因为readMode是cudaReadModeElementType，所以返回值类型还是unsigned char
     * @note 因为normalizedCoords是0，所以坐标系还是像素坐标系，未归一化
     * @note +0.5f是因为像素/texel的几何中心是在坐标(x+0.5, y+0.5)处。保证点采样和线性插值的一致性
     * @note addressMode是cudaAddressModeClamp，所以坐标超出边界时会被设置为边界值
     */
    unsigned char yv = tex2D<unsigned char>(texY, src_x + 0.5f, src_y + 0.5f);
    outY[x + y * crop_w] = yv;

    // UV planes: subsampled by 2 (I420)
    int uv_x = (crop_x / 2) + (x / 2);
    int uv_y = (crop_y / 2) + (y / 2);
    unsigned char uv = tex2D<unsigned char>(texU, uv_x + 0.5f, uv_y + 0.5f);
    unsigned char vv = tex2D<unsigned char>(texV, uv_x + 0.5f, uv_y + 0.5f);

    int out_uv_w = crop_w / 2;
    int out_uv_x = x / 2;
    int out_uv_y = y / 2;
    outU[out_uv_x + out_uv_y * out_uv_w] = uv;
    outV[out_uv_x + out_uv_y * out_uv_w] = vv;
}

void cropYUV420_using_cudaArray(const unsigned char* h_y, const unsigned char* h_u, const unsigned char* h_v,
                                int width, int height, int crop_x, int crop_y, int crop_w, int crop_h,
                                unsigned char* h_out_y, unsigned char* h_out_u, unsigned char* h_out_v)
{
    // 创建通道格式描述符，这里yuv各分量均为8位无符号整数
    cudaChannelFormatDesc chDesc = cudaCreateChannelDesc(8,0,0,0,cudaChannelFormatKindUnsigned);

    // allocate cudaArrays
    cudaArray_t arrY, arrU, arrV;
    CHECK_CUDA(cudaMallocArray(&arrY, &chDesc, width, height));
    CHECK_CUDA(cudaMallocArray(&arrU, &chDesc, width/2, height/2));
    CHECK_CUDA(cudaMallocArray(&arrV, &chDesc, width/2, height/2));

    /**
     * @brief 将主机内存中src指向的矩阵(height行，每行width字节)复制到cudaArray中。
     *        cudaArray的地址为dst，并从第hOffset行、第wOffset字节开始写入数据(左上角开始)。
     * @param dst 目标cudaArray
     * @param woffest 偏移量
     * @param hoffest 偏移量
     * @param src 源主机内存指针
     * @param spitch 矩阵在源内存中的行跨度（以字节为单位），可能包含padding
     * @param width 矩阵的宽度（以字节为单位）
     * @param height 矩阵的高度
     * @param kind 复制方向（从主机到设备）
     */
    CHECK_CUDA(cudaMemcpy2DToArray(arrY, 0, 0, h_y, width, width, height, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy2DToArray(arrU, 0, 0, h_u, width/2, width/2, height/2, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy2DToArray(arrV, 0, 0, h_v, width/2, width/2, height/2, cudaMemcpyHostToDevice));

    /**
     * @brief 描述了需要纹理的资源类型和相关信息
     *
     * struct cudaResourceDesc {
     *     enum cudaResourceType resType; // 需要进行纹理的资源类型（如Array，Linear等）
     *     union {
     *         struct {
     *             cudaArray_t array; // 指向cudaArray的指针
     *         } array;
     *         struct {
     *           cudaMipmappedArray_t mipmap; // 指向cudaMipmappedArray的指针
     *         } mipmap;
     *         struct {
     *             void* devPtr; // 指向线性内存的指针
     *             struct cudaChannelFormatDesc desc; // 线性内存的通道格式描述符
     *             size_t sizeInBytes; // 线性内存的大小（以字节为单位）
     *         } linear;
     *         struct {
     *             void* devPtr; // 指向立方体纹理内存的指针
     *             struct cudaChannelFormatDesc desc; // 立方体纹理的通道格式描述符
     *             size_t width; // 立方体纹理的宽度
     *             size_t height; // 立方体纹理的高度
     *             size_t pitchInBytes; // 立方体纹理的行跨度（以字节为单位）
     *         } pitch2D;
     *     }res;
     * };
     */
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;

    /**
     * @brief 描述了纹理对象的采样和访问方式
     *
     * struct cudaTextureDesc {
     *     // 纹理地址模式，决定纹理坐标被映射到0-1之外的情况，每个维度一个：
     *     // Repat重复纹理，Clamp使用临界值，Mirror镜像重复，Border使用定值
     *     enum cudaTextureAddressMode addressMode[3];
     *     // 决定了获取纹理数据时的过滤方式（如点采样、线性插值等）
     *     enum cudaTextureFilterMode filterMode;
     *     // 整形数据是否表示成浮点数
     *     enum cudaTextureReadMode readMode;
     *     int sRGB; // 是否使用sRGB颜色空间
     *     ... // 其他纹理描述符字段
     *     int normalizedCoords; // 是否使用归一化坐标（0表示使用整数坐标，1表示使用归一化坐标）
     * };
     */
    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0; // use integer coords

    /**
     * @brief 纹理对象
     */
    cudaTextureObject_t texY=0, texU=0, texV=0;
    // Y
    resDesc.res.array.array = arrY;
    CHECK_CUDA(cudaCreateTextureObject(&texY, &resDesc, &texDesc, nullptr));
    // U
    resDesc.res.array.array = arrU;
    CHECK_CUDA(cudaCreateTextureObject(&texU, &resDesc, &texDesc, nullptr));
    // V
    resDesc.res.array.array = arrV;
    CHECK_CUDA(cudaCreateTextureObject(&texV, &resDesc, &texDesc, nullptr));

    // allocate device linear outputs
    unsigned char *d_outY=nullptr, *d_outU=nullptr, *d_outV=nullptr;
    size_t outY_bytes = (size_t)crop_w * crop_h;
    size_t outUV_bytes = (size_t)(crop_w/2) * (crop_h/2);
    CHECK_CUDA(cudaMalloc(&d_outY, outY_bytes));
    CHECK_CUDA(cudaMalloc(&d_outU, outUV_bytes));
    CHECK_CUDA(cudaMalloc(&d_outV, outUV_bytes));

    // launch kernel
    dim3 block(16,16);
    dim3 grid((crop_w + block.x - 1)/block.x, (crop_h + block.y - 1)/block.y);
    cropYUV_kernel<<<grid, block>>>(texY, texU, texV, d_outY, d_outU, d_outV,
                                    width, height, crop_x, crop_y, crop_w, crop_h);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // copy back
    CHECK_CUDA(cudaMemcpy(h_out_y, d_outY, outY_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_out_u, d_outU, outUV_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_out_v, d_outV, outUV_bytes, cudaMemcpyDeviceToHost));

    // cleanup
    CHECK_CUDA(cudaDestroyTextureObject(texY));
    CHECK_CUDA(cudaDestroyTextureObject(texU));
    CHECK_CUDA(cudaDestroyTextureObject(texV));
    CHECK_CUDA(cudaFreeArray(arrY));
    CHECK_CUDA(cudaFreeArray(arrU));
    CHECK_CUDA(cudaFreeArray(arrV));
    CHECK_CUDA(cudaFree(d_outY));
    CHECK_CUDA(cudaFree(d_outU));
    CHECK_CUDA(cudaFree(d_outV));
}

void cropYUV420() {
    std::string input_yuv = "/mnt/workspace/cgz_workspace/Exercise/camera_example/input/650_yuv420p.yuv";
    std::string output_yuv = "/mnt/workspace/cgz_workspace/Exercise/camera_example/output/650_yuv420p_crop.yuv";
    const int width = 1080;
    const int height = 1920;
    const int crop_x = 100;
    const int crop_y = 200;
    const int crop_w = 640;
    const int crop_h = 480;
    size_t frame_size = (size_t)width * height * 3 / 2; // YUV420p
    size_t y_size = (size_t)width * height;
    size_t uv_size = (size_t)(width / 2) * (height / 2);
    size_t crop_y_size = (size_t)crop_w * crop_h;
    size_t crop_uv_size = (size_t)(crop_w / 2) * (crop_h / 2);
    std::ifstream infile(input_yuv, std::ios::binary|std::ios::in|std::ios::ate);
    if(!infile.is_open()) {
        std::cerr << "Failed to open input YUV file: " << input_yuv << std::endl;
        return;
    }
    size_t file_size = infile.tellg();
    if(file_size < frame_size) {
        std::cerr << "Input YUV file size is smaller than expected frame size." << std::endl;
        return;
    }
    infile.seekg(0, std::ios::beg);
    std::vector<unsigned char> h_y(y_size);
    std::vector<unsigned char> h_u(uv_size);
    std::vector<unsigned char> h_v(uv_size);
    infile.read(reinterpret_cast<char*>(h_y.data()), y_size);
    infile.read(reinterpret_cast<char*>(h_u.data()), uv_size);
    infile.read(reinterpret_cast<char*>(h_v.data()), uv_size);
    infile.close();
    std::vector<unsigned char> h_out_y(crop_y_size);
    std::vector<unsigned char> h_out_u(crop_uv_size);
    std::vector<unsigned char> h_out_v(crop_uv_size);
    cropYUV420_using_cudaArray(
        h_y.data(), h_u.data(), h_v.data(),
        width, height,
        crop_x, crop_y, crop_w, crop_h,
        h_out_y.data(), h_out_u.data(), h_out_v.data()
    );
    std::ofstream outfile(output_yuv, std::ios::binary|std::ios::out);
    if(!outfile.is_open()) {
        std::cerr << "Failed to open output YUV file: " << output_yuv << std::endl;
        return;
    }
    outfile.write(reinterpret_cast<char*>(h_out_y.data()), crop_y_size);
    outfile.write(reinterpret_cast<char*>(h_out_u.data()), crop_uv_size);
    outfile.write(reinterpret_cast<char*>(h_out_v.data()), crop_uv_size);
    outfile.close();
}

int main(int argc, char* argv[]) {
    std::cout << "===================== cuda_stream_event =====================" << std::endl;
    cuda_stream_event();
    std::cout << "===================== cuda_memcpy_async =====================" << std::endl;
    cuda_memcpy_async();
    std::cout << "===================== cropYUV420 =====================" << std::endl;
    cropYUV420();
    return 0;
}
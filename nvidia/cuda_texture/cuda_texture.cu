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
    std::cout << "Channel Format Desc: x=" << chDesc.x << ", y=" << chDesc.y
              << ", z=" << chDesc.z << ", w=" << chDesc.w
              << ", f=" << static_cast<int>(chDesc.f) << std::endl;

    // allocate cudaArrays
    cudaArray_t arrY, arrU, arrV;
    CHECK_CUDA(cudaMallocArray(&arrY, &chDesc, width, height));
    CHECK_CUDA(cudaMallocArray(&arrU, &chDesc, width/2, height/2));
    CHECK_CUDA(cudaMallocArray(&arrV, &chDesc, width/2, height/2));

    size_t dst_pitch = width * sizeof(unsigned char); // 每行的字节数
    size_t dst_width = width * sizeof(unsigned char); // 复制的宽度（以字节为单位）
    size_t dst_height = height;                        // 复制的高度
    unsigned char* dst_ptr = nullptr;              // 目标内存指针
    /**
     * @brief 分配二维数组的内存，并返回指向分配内存的指针和行跨度
     * @param devPtr 指向分配内存的指针
     * @param pitch 返回分配内存的行跨度（以字节为单位）
     * @param width 分配内存的宽度（以字节为单位）
     * @param height 分配内存的高度
     *
     * @note 返回的pitch可能大于width，以满足内存对齐要求，提高内存访问效率
     * @note pitch也用于计算二维数组数据的地址：dst_ptr = base_ptr + row * pitch + col
     */
    CHECK_CUDA(cudaMallocPitch(&dst_ptr, &dst_pitch, dst_width, dst_height));
    std::cout << "Allocated pitched memory: ptr=" << static_cast<void*>(dst_ptr)
              << ", pitch=" << dst_pitch
              << ", width=" << dst_width
              << ", height=" << dst_height << std::endl;

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
     * @brief 从src指向的cudaArray的hOffset行、wOffset字节开始读取数据(左上角开始)，
     *        并将读取到的矩阵(height行，每行width字节)复制到线性的目标内存dst中。
     * @param dst 目标线性内存指针
     * @param dpitch 目标内存的行跨度（以字节为单位）
     * @param src 源cudaArray地址
     * @param woffest 偏移量(以字节为单位)
     * @param hoffest 偏移量
     * @param width 矩阵的宽度（以字节为单位）
     * @param height 矩阵的高度
     * @param kind 复制方向（从设备到设备）
     * @param stream CUDA流，用于异步操作，不指定则为默认流
     *
     * @note 本来想使用cudaMemcpyFast2DFromArrayAsync，但是当前机器cuda不支持
     */
    CHECK_CUDA(cudaMemcpy2DFromArrayAsync(dst_ptr, dst_pitch, arrY, 0, 0, dst_width, dst_height, cudaMemcpyDeviceToDevice));

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
     * typedef __device_builtin__ unsigned long long cudaTextureObject_t
     */
    cudaTextureObject_t texY=0, texU=0, texV=0;

    // Y
    resDesc.res.array.array = arrY;
    /**
     * @brief 创建一个纹理对象
     * @param pTexObject 返回创建的纹理对象
     * @param pResDesc 纹理资源描述符，描述了需要进行纹理的数据源和相关信息
     * @param pTexDesc 纹理描述符，描述了纹理对象的采样和访问方式
     * @param pResViewDesc 资源视图描述符，描述了纹理对象的资源视图信息（如mipmap级别、数组层等），如果不使用则为nullptr
     */
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
    CHECK_CUDA(cudaFree(dst_ptr));
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
    std::cout << "Input YUV file: " << input_yuv << std::endl;
    std::cout << "Output YUV file: " << output_yuv << std::endl;
    std::cout << "Frame dimensions: " << width << "x" << height << std::endl;
    std::cout << "Crop rectangle: (" << crop_x << ", " << crop_y << ", " << crop_w << ", " << crop_h << ")" << std::endl;
    std::cout << "Input YUV file size: " << file_size << " bytes" << std::endl;
    std::cout << "Expected frame size: " << frame_size << " bytes" << std::endl;

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
    std::cout << "===================== cropYUV420 =====================" << std::endl;
    cropYUV420();
    
    return 0;
}
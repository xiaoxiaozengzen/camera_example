#include <iostream>
#include <string>
#include <sys/types.h>
#include <vector>
#include <fstream>

#include <cuda_runtime.h>

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

/**
 * 原始图像数据可能是yuv，
 * 但是输入给模型的图像一般转成NCHW格式的RGB图像，
 */

/**
 * struct __device_builtin__ uchar3 {
 *     unsigned char x;
 *     unsigned char y;
 *     unsigned char z;
 * };
 */

/**
 * 处理图像的时候，kernel的block跟grid的大小设置：
 * 1. block的大小尽量是32的倍数，因为GPU的线程束（warp）是32个线程一起执行的。
 *    常用128，256，512等大小。
 *    不要超过1024，因为每个block最多只能有1024个线程。
 *    常用例子：blockDim.x = 64，blockDim.y = 4，blockDim.z = 1。
 * 2. grid的大小根据图像的尺寸和block的大小计算：
 *    gridDim.x = (image_width + blockDim.x - 1) / blockDim.x;
 *    gridDim.y = (image_height + blockDim.y - 1) / blockDim.y;
 *    gridDim.z = batch_size; // 如果有批处理的话
 */

/*************************************** undistort ***************************************/

__device__ void __forceinline__ GetRGBPixel(uint8_t* yuv_data, int32_t x, int32_t y, int32_t height, int32_t width, uchar3& rgb_pixel) {
    int y_index = y * width + x; // Y分量的索引
    int uv_index = (height * width) + (y / 2) * width + (x / 2) * 2; // UV分量的索引，假设是NV12格式，UV分量交错存储

    uint8_t Y = yuv_data[y_index];
    uint8_t U = yuv_data[uv_index + 0]; // U分量
    uint8_t V = yuv_data[uv_index + 1]; // V分量

    int32_t Y_Item = 0;
    int32_t DR = 0;
    int32_t DG = 0;
    int32_t DB = 0;

    /**
     * YUV转RGB的计算公式(ITU-R BT.601标准)：
     * R = 1.164 * (Y - 16) + 1.596 * (V - 128)
     * G = 1.164 * (Y - 16) - 0.813 * (V - 128) - 0.391 * (U - 128)
     * B = 1.164 * (Y - 16) + 2.018 * (U - 128)
     *
     * 下面的YUV转RGB的计算公式采用了整数近似+位移缩放的高效实现方式，
     * 目的是为了在GPU上高效计算，避免使用浮点数运算
     * 使用缩放因子2^20来近似浮点数乘法：
     * 1.164近似1220542/2^20，1.596近似1673527/2^20，0.813近似852492/2^20
     * 0.391近似411041/2^20，2.018近似2114977/2^20
     *
     */
    Y_Item = 1220542 * ((Y - 16)<=0? 0 : (Y - 16)>255? 255 : (Y - 16)); // Y分量的缩放和偏移，近似乘以1.164
    DR = 1673527 * (V - 128);
    DG = 852492 * (V - 128) + 411041 * (U - 128);
    DB = 2114977 * (U - 128);

    // 1<<19是为了实现四舍五入，右移20位相当于除以2^20，得到最终的RGB值
    int16_t R = (int16_t)((Y_Item + DR + (1 << 19)) >> 20);
    int16_t G = (int16_t)((Y_Item - DG + (1 << 19)) >> 20);
    int16_t B = (int16_t)((Y_Item + DB + (1 << 19)) >> 20);

    rgb_pixel.x = R < 0 ? 0 : (R > 255 ? 255 : R); // 将结果限制在0-255范围内
    rgb_pixel.y = G < 0 ? 0 : (G > 255 ? 255 : G);
    rgb_pixel.z = B < 0 ? 0 : (B > 255 ? 255 : B);
}

__device__ void __forceinline__ UndistortPoint(float& u_norm, float& v_norm, const float* calib_param) {
    // calib_param: [fx, fy, cx, cy, k1, k2, k3, k4]
    float k1 = calib_param[4];
    float k2 = calib_param[5];
    float k3 = calib_param[6];
    float k4 = calib_param[7];

    // 基于OpenCV的畸变模型，将归一化坐标从理想(去畸变)映射到实际(有畸变)坐标的过程：
    // 1.计算半径
    float r = sqrt(u_norm * u_norm + v_norm * v_norm);
    // 2.计算极角：点与光轴的夹角
    float theta = atan(r);
    // 3.计算极角的高次幂
    float theta2 = theta * theta;
    float theta4 = theta2 * theta2;
    float theta6 = theta4 * theta2;
    float theta8 = theta4 * theta4;
    // 4.计算修正后的极角
    float theta_d = theta * (1 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8);
    // 5.计算修正后的归一化坐标
    u_norm = u_norm * (theta_d / r);
    v_norm = v_norm * (theta_d / r);
}

template<bool need_undistort>
__device__ void __forceinline__ GetDistortedPoint(const float* calib_param, const int32_t& u_src_undistorted, const int32_t& v_src_undistorted, int32_t& u_src_distorted, int32_t& v_src_distorted);

template<>
__device__ void __forceinline__ GetDistortedPoint<true>(const float* calib_param, const int32_t& u_src_undistorted, const int32_t& v_src_undistorted, int32_t& u_src_distorted, int32_t& v_src_distorted) {
    // calib_param: [fx, fy, cx, cy, k1, k2, k3, k4]
    float fx = calib_param[0];
    float fy = calib_param[1];
    float cx = calib_param[2];
    float cy = calib_param[3];
    float fx_rd = calib_param[8];
    float fy_rd = calib_param[9];
    float cx_offset = calib_param[10];
    float cy_offset = calib_param[11];

    // 归一化:以相机光心为原点，焦距为单位，将像素坐标转换为归一化坐标
    // 为什么归一化：归一化可以将不同分辨率和不同焦距的图像统一到相同的尺度，便于后续的畸变校正计算
    float u_norm = (u_src_undistorted - cx + cx_offset) / fx_rd;
    float v_norm = (v_src_undistorted - cy + cy_offset) / fy_rd;

    UndistortPoint(u_norm, v_norm, calib_param);

    // 原始畸变图像上需要采样的坐标
    u_src_distorted = round(fx * u_norm + cx);
    v_src_distorted = round(fy * v_norm + cy);
}

template<>
__device__ void __forceinline__ GetDistortedPoint<false>(const float* calib_param, const int32_t& u_src_undistorted, const int32_t& v_src_undistorted, int32_t& u_src_distorted, int32_t& v_src_distorted) {
    // 如果不需要畸变校正，直接返回输入的坐标
    u_src_distorted = u_src_undistorted;
    v_src_distorted = v_src_undistorted;
}

template<bool need_undistort>
__device__ void __forceinline__ GetRGBDataKernel(uint8_t* yuv_data, const int32_t* roi_param, const float* calib_param, const int32_t* dst_image_size, const bool& keep_shape,int u, int v, uchar3& rgb_data) {
    int32_t src_image_width = roi_param[0];
    int32_t src_image_height = roi_param[1];
    int32_t roi_x = roi_param[2];
    int32_t roi_y = roi_param[3];
    int32_t roi_width = roi_param[4];
    int32_t roi_height = roi_param[5];
    int32_t dst_image_width = dst_image_size[0];
    int32_t dst_image_height = dst_image_size[1];

    // 将目标图像的坐标(u, v)映射回原始畸变图像的坐标
    // 可以这样理解：
    // u/dst_image_width = (u_src_undistorted - roi_x) / roi_width
    // v/dst_image_height = (v_src_undistorted - roi_y) / roi_height
    int32_t u_src_undistorted = (u * 1.0f * roi_width / dst_image_width) + roi_x;
    int32_t v_src_undistorted = (v * 1.0f * roi_height / dst_image_height) + roi_y;

    int valid_width = dst_image_width;
    int valid_height = dst_image_height;

    /**
     * 目标图像和ROI区域的宽高比是否要一致：
     * 1.有些任务(例如目标检测，分割)需要输入的图片等比例缩放+填充黑边。这就需要keep_shape=true,保证内容不形变
     * 2.有些任务(例如全景拼接)需要图片拉伸填满，不关心形变。这时候keep_shape=false
     */
    if(keep_shape) {
        // 等比例缩放的时候的最小缩放因子，保证ROI区域图能够完全放进目标图像中
        float scale = min((float)dst_image_width / roi_width, (float)dst_image_height / roi_height);
        valid_width = round(roi_width * scale);  // roi区域能够填充目标图像的宽，即有效区域的宽
        valid_height = round(roi_height * scale);  // roi区域能够填充目标图像的高，即有效区域的高
        // 等比例缩放关系：u_dst = (u_src - roi_x) * scale
        u_src_undistorted = (u * 1.0f / scale) + roi_x; // 目标图像上的坐标映射到ROI区域图上的坐标，有可能超过ROI区域的范围
        v_src_undistorted = (v * 1.0f / scale) + roi_y;
    }

    int u_src_distorted = -1;
    int v_src_distorted = -1;
    GetDistortedPoint<need_undistort>(calib_param, u_src_undistorted, v_src_undistorted, u_src_distorted, v_src_distorted);
    if(u < valid_width && v < valid_height
        && u_src_distorted >=0 && v_src_distorted >=0
        && u_src_distorted < src_image_width && v_src_distorted < src_image_height)
    {
        // 如果目标图像上的坐标在有效区域内，并且映射回原始图像上的坐标也在原始图像的范围内，那么这个像素点就有有效的输入图像数据，可以根据映射回原始图像上的坐标从输入图像数据中采样得到RGB值
        GetRGBPixel(yuv_data, u_src_distorted, v_src_distorted, src_image_height, src_image_width, rgb_data);
    } else {
        // 如果目标图像上的坐标超过了有效区域，或者映射回原始图像上的坐标超过了原始图像的范围，那么这个像素点就没有有效的输入图像数据了，可以设置为0或者其他默认值
        rgb_data.x = 0;
        rgb_data.y = 0;
        rgb_data.z = 0;
    }
    
}

template<bool need_undistort>
__global__ void UndistortKernel(uint8_t* yuv_data, const int32_t* roi_param, const float* calib_param, const int32_t* dst_image_size, const bool* keep_shape, uint8_t* output_rgb_data) {
    int u = blockIdx.x * blockDim.x + threadIdx.x; // 目标图像上的x坐标
    int v = blockIdx.y * blockDim.y + threadIdx.y; // 目标图像上的y坐标
    int dst_image_width = dst_image_size[0];
    int dst_image_height = dst_image_size[1];
    if (u >= dst_image_width || v >= dst_image_height) {
        return; // 超出目标图像范围的线程不处理
    }

    uchar3 rgb_data;
    GetRGBDataKernel<need_undistort>(yuv_data, roi_param, calib_param, dst_image_size, *keep_shape, u, v, rgb_data);

    // 将得到的RGB值写入输出图像数据中，假设输出图像数据是按照NCHW格式存储的
    int image_size = dst_image_width * dst_image_height;
    int dst_index = v * dst_image_width + u; // 目标图像上的坐标对应的索引
    output_rgb_data[dst_index] = rgb_data.x; // R分量
    output_rgb_data[image_size + dst_index] = rgb_data.y; // G分量
    output_rgb_data[2 * image_size + dst_index] = rgb_data.z; // B分量
}

void undistort() {
    // 输入图像数据，假设是yuv格式
    std::string input_image_path = "/mnt/workspace/cgz_workspace/Exercise/camera_example/input/frontwide_3840_2048_nv12.yuv";
    int image_width = 3840;
    int image_height = 2048;
    int yuv_size = image_width * image_height * 3 / 2; // nv12格式的图像大小
    std::vector<uint8_t> yuv_data(yuv_size);
    std::ifstream input_file(input_image_path, std::ios::binary|std::ios::ate);
    if (!input_file.is_open()) {
        std::cerr << "Failed to open input image file: " << input_image_path << std::endl;
        return;
    }
    std::size_t file_size = input_file.tellg();
    if (file_size != yuv_size) {
        std::cerr << "Input image file size does not match expected size: " << file_size << " vs " << yuv_size << std::endl;
        return;
    }
    input_file.seekg(0, std::ios::beg);
    input_file.read(reinterpret_cast<char*>(yuv_data.data()), yuv_size);
    input_file.close();
    std::cout << "Input image size: " << file_size << " bytes" << std::endl;

    // 设置输入图像数据的GPU内存
    uint8_t* yuv_data_gpu = nullptr;
    CHECK_CUDA(cudaMalloc(&yuv_data_gpu, yuv_size));
    CHECK_CUDA(cudaMemcpy(yuv_data_gpu, yuv_data.data(), yuv_size, cudaMemcpyHostToDevice));

    // 设置roi
    int32_t roi_param[6] = {image_width, image_height, 0, 0, image_width, image_height}; // roi参数，假设是全图
    int32_t* roi_param_gpu = nullptr;
    CHECK_CUDA(cudaMalloc(&roi_param_gpu, sizeof(roi_param)));
    CHECK_CUDA(cudaMemcpy(roi_param_gpu, roi_param, sizeof(roi_param), cudaMemcpyHostToDevice));
    std::cout << "ROI: image_width=" << roi_param[0] << ", image_height=" << roi_param[1] << std::endl;
    std::cout << "ROI: x=" << roi_param[2] << ", y=" << roi_param[3] << ", width=" << roi_param[4] << ", height=" << roi_param[5] << std::endl;

    // 设置输出图像的尺寸
    int32_t dst_image_size[2] = {image_width / 2, image_height / 2}; // 输出图像的尺寸
    int32_t* dst_image_size_gpu = nullptr;
    CHECK_CUDA(cudaMalloc(&dst_image_size_gpu, sizeof(dst_image_size)));
    CHECK_CUDA(cudaMemcpy(dst_image_size_gpu, dst_image_size, sizeof(dst_image_size), cudaMemcpyHostToDevice));
    std::cout << "Output image size: width=" << dst_image_size[0] << ", height=" << dst_image_size[1] << std::endl;

    // 设置输出图像数据的GPU内存
    uint8_t* output_rgb_data_cpu = nullptr; // 输出图像数据的CPU内存
    output_rgb_data_cpu = new uint8_t[dst_image_size[0] * dst_image_size[1] * 3]; // 输出图像数据的大小，假设是RGB格式
    memset(output_rgb_data_cpu, 0, dst_image_size[0] * dst_image_size[1] * 3); // 初始化输出图像数据
    uint8_t* output_rgb_data_gpu = nullptr; // 输出图像数据的GPU内存
    CHECK_CUDA(cudaMalloc(&output_rgb_data_gpu, dst_image_size[0] * dst_image_size[1] * 3));
    CHECK_CUDA(cudaMemcpy(output_rgb_data_gpu, output_rgb_data_cpu, dst_image_size[0] * dst_image_size[1] * 3, cudaMemcpyHostToDevice));

    // 设置相机内参和畸变参数
    float calib_param[12] = {1906.6, 1906.18, 1923.26, 1022.45, -0.0299548, -0.00364585, -0.00155829, 0.00104736}; // fx, fy, cx, cy, k1, k2, k3, k4
    float fx_scale = 1.0;
    float fy_scale = 1.0;
    float cx_offset = 0.0;
    float cy_offset = 0.0;
    calib_param[8] = calib_param[0] / fx_scale;
    calib_param[9] = calib_param[1] / fy_scale;
    calib_param[10] = cx_offset;
    calib_param[11] = cy_offset;
    float* calib_param_gpu = nullptr;
    CHECK_CUDA(cudaMalloc(&calib_param_gpu, sizeof(calib_param)));
    CHECK_CUDA(cudaMemcpy(calib_param_gpu, calib_param, sizeof(calib_param), cudaMemcpyHostToDevice));
    std::cout << "Camera intrinsics: fx=" << calib_param[0] << ", fy=" << calib_param[1] << ", cx=" << calib_param[2] << ", cy=" << calib_param[3] << std::endl;
    std::cout << "Distortion coefficients: k1=" << calib_param[4] << ", k2=" << calib_param[5] << ", k3=" << calib_param[6] << ", k4=" << calib_param[7] << std::endl;
    std::cout << "fx_scale=" << fx_scale << ", fy_scale=" << fy_scale << ", cx_offset=" << cx_offset << ", cy_offset=" << cy_offset << std::endl;
    std::cout << "fx_rd=" << calib_param[8] << ", fy_rd=" << calib_param[9] << ", cx_offset=" << calib_param[10] << ", cy_offset=" << calib_param[11] << std::endl;

    // 设置是否保持图像内容的宽高比
    bool keep_shape = false;
    bool* keep_shape_gpu = nullptr;
    CHECK_CUDA(cudaMalloc(&keep_shape_gpu, sizeof(bool)));
    CHECK_CUDA(cudaMemcpy(keep_shape_gpu, &keep_shape, sizeof(bool), cudaMemcpyHostToDevice));
    std::cout << "Keep shape: " << (keep_shape ? "true" : "false") << std::endl;

    // 启动kernel进行图像去畸变和格式转换
    dim3 block_dim(64, 4); // 每个block有64*4=256个线程
    dim3 grid_dim((dst_image_size[0] + block_dim.x - 1) / block_dim.x, (dst_image_size[1] + block_dim.y - 1) / block_dim.y); // 根据输出图像的尺寸计算需要多少个block
    std::cout << "Block dim: (" << block_dim.x << ", " << block_dim.y << ", " << block_dim.z << ")" << std::endl;
    std::cout << "Grid dim: (" << grid_dim.x << ", " << grid_dim.y << ", " << grid_dim.z << ")" << std::endl;
    UndistortKernel<true><<<grid_dim, block_dim>>>(yuv_data_gpu, roi_param_gpu, calib_param_gpu, dst_image_size_gpu, keep_shape_gpu, output_rgb_data_gpu);
    CHECK_CUDA(cudaDeviceSynchronize()); // 等待kernel执行完成  

    // 将处理后的图像数据从GPU内存复制回CPU内存
    CHECK_CUDA(cudaMemcpy(output_rgb_data_cpu, output_rgb_data_gpu, dst_image_size[0] * dst_image_size[1] * 3, cudaMemcpyDeviceToHost));
    std::string output_image_path = "/mnt/workspace/cgz_workspace/Exercise/camera_example/output/undistort_frontwide_1920_1024.rgb"; // 输出图像的路径，假设是rgb格式
    std::ofstream output_file(output_image_path, std::ios::binary);
    if (!output_file.is_open()) {
        std::cerr << "Failed to open output image file: " << output_image_path << std::endl;
        return;
    }
    output_file.write(reinterpret_cast<char*>(output_rgb_data_cpu), dst_image_size[0] * dst_image_size[1] * 3);
    output_file.close();
    std::cout << "Output image saved to: " << output_image_path << std::endl;

    // 将处理后的图像数据按照nv12格式保存，方便后续验证
    std::string output_nv12_image_path = "/mnt/workspace/cgz_workspace/Exercise/camera_example/output/undistort_frontwide_1920_1024_nv12.yuv";
    std::ofstream output_nv12_file(output_nv12_image_path, std::ios::binary);
    if (!output_nv12_file.is_open()) {
        std::cerr << "Failed to open output nv12 image file: " << output_nv12_image_path << std::endl;
        return;
    }
    // 将RGB格式的图像数据转换为NV12格式并保存
    std::vector<uint8_t> nv12_data(dst_image_size[0] * dst_image_size[1] * 3 / 2); // NV12格式的图像大小
    for (int v = 0; v < dst_image_size[1]; ++v) {
        for (int u = 0; u < dst_image_size[0]; ++u) {
            int dst_index = v * dst_image_size[0] + u;
            uint8_t R = output_rgb_data_cpu[dst_index]; // R分量
            uint8_t G = output_rgb_data_cpu[dst_image_size[0] * dst_image_size[1] + dst_index]; // G分量
            uint8_t B = output_rgb_data_cpu[2 * dst_image_size[0] * dst_image_size[1] + dst_index]; // B分量
            // RGB转YUV的计算公式(ITU-R BT.601标准)：
            // Y = 0.299 * R + 0.587 * G + 0.114 * B
            // U = -0.169 * R - 0.331 * G + 0.5 * B + 128
            // V = 0.5 * R - 0.419 * G - 0.081 * B + 128
            uint8_t Y = static_cast<uint8_t>(0.299 * R + 0.587 * G + 0.114 * B);
            uint8_t U = static_cast<uint8_t>(-0.169 * R - 0.331 * G + 0.5 * B + 128);
            uint8_t V = static_cast<uint8_t>(0.5 * R - 0.419 * G - 0.081 * B + 128);
            nv12_data[v * dst_image_size[0] + u] = Y; // Y分量
            if (v % 2 == 0 && u % 2 == 0) {
                int uv_index = (dst_image_size[0] * dst_image_size[1]) + (v / 2) * dst_image_size[0] + (u / 2) * 2; // UV分量的索引，假设是NV12格式，UV分量交错存储
                nv12_data[uv_index + 0] = U; // U分量
                nv12_data[uv_index + 1] = V; // V分量
            }
        }
    }
    output_nv12_file.write(reinterpret_cast<char*>(nv12_data.data()), nv12_data.size());
    output_nv12_file.close();
    std::cout << "Output NV12 image saved to: " << output_nv12_image_path << std::endl;

    int rectangle_x = 1520;
    int rectangle_y = 20;
    int rectangle_width = 100;
    int rectangle_height = 100;
    for (int v = rectangle_y; v < rectangle_y + rectangle_height; ++v) {
        for (int u = rectangle_x; u < rectangle_x + rectangle_width; ++u) {
            if (v < dst_image_size[1] && u < dst_image_size[0]) {
                int dst_index = v * dst_image_size[0] + u;
                nv12_data[dst_index] = 128; // 将Y分量设置为128，形成一个灰色的矩形区域
                if (v % 2 == 0 && u % 2 == 0) {
                    int uv_index = (dst_image_size[0] * dst_image_size[1]) + (v / 2) * dst_image_size[0] + (u / 2) * 2; // UV分量的索引，假设是NV12格式，UV分量交错存储
                    nv12_data[uv_index + 0] = 128; // 将U分量设置为128
                    nv12_data[uv_index + 1] = 128; // 将V分量设置为128，形成一个灰色的矩形区域
                }
            }
        }
    }
    std::string output_pixelated_nv12_image_path = "/mnt/workspace/cgz_workspace/Exercise/camera_example/output/undistort_frontwide_1920_1024_nv12_pixelated.yuv";
    std::ofstream output_pixelated_nv12_file(output_pixelated_nv12_image_path, std::ios::binary);
    if (!output_pixelated_nv12_file.is_open()) {
        std::cerr << "Failed to open output pixelated nv12 image file: " << output_pixelated_nv12_image_path << std::endl;
        return;
    }
    output_pixelated_nv12_file.write(reinterpret_cast<char*>(nv12_data.data()), nv12_data.size());
    output_pixelated_nv12_file.close();
    std::cout << "Output pixelated NV12 image saved to: " << output_pixelated_nv12_image_path << std::endl;

    // 清理GPU内存
    CHECK_CUDA(cudaFree(yuv_data_gpu));
    CHECK_CUDA(cudaFree(output_rgb_data_gpu));
    CHECK_CUDA(cudaFree(roi_param_gpu));
    CHECK_CUDA(cudaFree(dst_image_size_gpu));
    CHECK_CUDA(cudaFree(calib_param_gpu));
    CHECK_CUDA(cudaFree(keep_shape_gpu));
    // 清理CPU内存
    delete[] output_rgb_data_cpu;
}

/*************************************** pixlate ***************************************/

__global__ void GetSrcRectangleKernel(const float* calib_param, const int32_t* roi_param, const int32_t* dst_image_size, const int32_t* dst_rectangle, int32_t* src_rectangle) {
    // 输出图像上的四个顶点坐标
    int32_t lu_u = dst_rectangle[0];
    int32_t lu_v = dst_rectangle[1];
    int32_t rd_u = dst_rectangle[2];
    int32_t rd_v = dst_rectangle[3];
    int32_t ld_u = dst_rectangle[4];
    int32_t ld_v = dst_rectangle[5];
    int32_t ru_u = dst_rectangle[6];
    int32_t ru_v = dst_rectangle[7];

    // roi参数
    int32_t roi_x = roi_param[2];
    int32_t roi_y = roi_param[3];
    int32_t roi_width = roi_param[4];
    int32_t roi_height = roi_param[5];
    int32_t dst_image_width = dst_image_size[0];
    int32_t dst_image_height = dst_image_size[1];


    // 将输出图像上的四个顶点坐标映射回原始畸变图像上的坐标
    GetDistortedPoint<true>(calib_param, (lu_u * 1.0f * roi_width / dst_image_width) + roi_x, (lu_v * 1.0f * roi_height / dst_image_height) + roi_y, src_rectangle[0], src_rectangle[1]);
    GetDistortedPoint<true>(calib_param, (rd_u * 1.0f * roi_width / dst_image_width) + roi_x, (rd_v * 1.0f * roi_height / dst_image_height) + roi_y, src_rectangle[2], src_rectangle[3]);
    GetDistortedPoint<true>(calib_param, (ld_u * 1.0f * roi_width / dst_image_width) + roi_x, (ld_v * 1.0f * roi_height / dst_image_height) + roi_y, src_rectangle[4], src_rectangle[5]);
    GetDistortedPoint<true>(calib_param, (ru_u * 1.0f * roi_width / dst_image_width) + roi_x, (ru_v * 1.0f * roi_height / dst_image_height) + roi_y, src_rectangle[6], src_rectangle[7]);
}
struct Point2F {
    float x;
    float y;
};

void pixlate() {
    // 设置相机内参和畸变参数
    float calib_param[12] = {1906.6, 1906.18, 1923.26, 1022.45, -0.0299548, -0.00364585, -0.00155829, 0.00104736}; // fx, fy, cx, cy, k1, k2, k3, k4
    float fx_scale = 1.0;
    float fy_scale = 1.0;
    float cx_offset = 0.0;
    float cy_offset = 0.0;
    calib_param[8] = calib_param[0] / fx_scale;
    calib_param[9] = calib_param[1] / fy_scale;
    calib_param[10] = cx_offset;
    calib_param[11] = cy_offset;
    int32_t image_width = 3840;
    int32_t image_height = 2048;
    int32_t roi_param[6] = {image_width, image_height, 0, 0, image_width, image_height}; // roi参数，假设是全图
    int32_t dst_image_size[2] = {image_width / 2, image_height / 2}; // 输出图像的尺寸

    float* calib_param_gpu = nullptr;
    CHECK_CUDA(cudaMalloc(&calib_param_gpu, sizeof(calib_param)));
    CHECK_CUDA(cudaMemcpy(calib_param_gpu, calib_param, sizeof(calib_param), cudaMemcpyHostToDevice));
    int32_t* roi_param_gpu = nullptr;
    CHECK_CUDA(cudaMalloc(&roi_param_gpu, sizeof(roi_param)));
    CHECK_CUDA(cudaMemcpy(roi_param_gpu, roi_param, sizeof(roi_param), cudaMemcpyHostToDevice));
    int32_t* dst_image_size_gpu = nullptr;
    CHECK_CUDA(cudaMalloc(&dst_image_size_gpu, sizeof(dst_image_size)));
    CHECK_CUDA(cudaMemcpy(dst_image_size_gpu, dst_image_size, sizeof(dst_image_size), cudaMemcpyHostToDevice));

    // 目标图像的一个矩形区域的四个顶点坐标
    Point2F lu = {1520.0f, 20.0f}; // 左上角坐标
    Point2F rd = {1620.0f, 120.0f}; // 右下角坐标
    Point2F ld = {1520.0f, 120.0f}; // 左下角坐标
    Point2F ru = {1620.0f, 20.0f}; // 右上角坐标
    int32_t dst_rectangle[8] = {static_cast<int32_t>(lu.x), static_cast<int32_t>(lu.y), static_cast<int32_t>(rd.x), static_cast<int32_t>(rd.y), static_cast<int32_t>(ld.x), static_cast<int32_t>(ld.y), static_cast<int32_t>(ru.x), static_cast<int32_t>(ru.y)};
    int32_t* dst_rectangle_gpu = nullptr;
    CHECK_CUDA(cudaMalloc(&dst_rectangle_gpu, sizeof(dst_rectangle)));
    CHECK_CUDA(cudaMemcpy(dst_rectangle_gpu, dst_rectangle, sizeof(dst_rectangle), cudaMemcpyHostToDevice));
    std::cout << "Target rectangle coordinates in the output image: " << std::endl;
    std::cout << "Left-upper corner: (" << dst_rectangle[0] << ", " << dst_rectangle[1] << ")" << std::endl;
    std::cout << "Right-lower corner: (" << dst_rectangle[2] << ", " << dst_rectangle[3] << ")" << std::endl;
    std::cout << "Left-lower corner: (" << dst_rectangle[4] << ", " << dst_rectangle[5] << ")" << std::endl;
    std::cout << "Right-upper corner: (" << dst_rectangle[6] << ", " << dst_rectangle[7] << ")" << std::endl;

    int32_t src_rectangle[8] = {0}; // 原始图像上对应的矩形区域的四个顶点坐标
    int32_t* src_rectangle_gpu = nullptr;
    CHECK_CUDA(cudaMalloc(&src_rectangle_gpu, sizeof(src_rectangle)));
    GetSrcRectangleKernel<<<1, 1>>>(calib_param_gpu, roi_param_gpu, dst_image_size_gpu, dst_rectangle_gpu, src_rectangle_gpu);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(src_rectangle, src_rectangle_gpu, sizeof(src_rectangle), cudaMemcpyDeviceToHost));
    std::cout << "Source rectangle coordinates in the original image corresponding to the target rectangle: " << std::endl;
    std::cout << "Left-upper corner: (" << src_rectangle[0] << ", " << src_rectangle[1] << ")" << std::endl;
    std::cout << "Right-lower corner: (" << src_rectangle[2] << ", " << src_rectangle[3] << ")" << std::endl;
    std::cout << "Left-lower corner: (" << src_rectangle[4] << ", " << src_rectangle[5] << ")" << std::endl;
    std::cout << "Right-upper corner: (" << src_rectangle[6] << ", " << src_rectangle[7] << ")" << std::endl;

    // 输入输出图像
    int src_rectangle_x = min(min(src_rectangle[0], src_rectangle[2]), min(src_rectangle[4], src_rectangle[6]));
    int src_rectangle_y = min(min(src_rectangle[1], src_rectangle[3]), min(src_rectangle[5], src_rectangle[7]));
    int src_rectangle_width = max(max(src_rectangle[0], src_rectangle[2]), max(src_rectangle[4], src_rectangle[6])) - src_rectangle_x;
    int src_rectangle_height = max(max(src_rectangle[1], src_rectangle[3]), max(src_rectangle[5], src_rectangle[7])) - src_rectangle_y;
    std::string input_image_path = "/mnt/workspace/cgz_workspace/Exercise/camera_example/input/frontwide_3840_2048_nv12.yuv";
    std::string output_image_path = "/mnt/workspace/cgz_workspace/Exercise/camera_example/output/undistort_frontwide_3840_2048_nv12_pixelated.yuv";
    std::ifstream input_file(input_image_path, std::ios::binary|std::ios::ate);
    if (!input_file.is_open()) {
        std::cerr << "Failed to open input image file: " << input_image_path << std::endl;
        return;
    }
    std::size_t file_size = input_file.tellg();
    int file_size_expected = image_width * image_height * 3 / 2; // nv12格式的图像大小
    if (file_size != file_size_expected) {
        std::cerr << "Input image file size does not match expected size: " << file_size << " vs " << file_size_expected << std::endl;
        return;
    }
    input_file.seekg(0, std::ios::beg);
    std::vector<uint8_t> image_data(file_size);
    input_file.read(reinterpret_cast<char*>(image_data.data()), file_size);
    input_file.close();

    for(int u = src_rectangle_x; u < src_rectangle_x + src_rectangle_width; ++u) {
        for(int v = src_rectangle_y; v < src_rectangle_y + src_rectangle_height; ++v) {
            if(u >= 0 && u < image_width && v >= 0 && v < image_height) {
                int yuv_index = v * image_width + u; // Y分量的索引
                // 将Y分量设置为128，形成一个灰色的像素点
                image_data[yuv_index] = 128;
                if (v % 2 == 0 && u % 2 == 0) {
                    int uv_index = (image_width * image_height) + (v / 2) * image_width + (u / 2) * 2; // UV分量的索引，假设是NV12格式，UV分量交错存储
                    image_data[uv_index + 0] = 128; // 将U分量设置为128
                    image_data[uv_index + 1] = 128; // 将V分量设置为128
                }
            }
        }
    }
    std::ofstream output_file(output_image_path, std::ios::binary);
    if (!output_file.is_open()) {
        std::cerr << "Failed to open output image file: " << output_image_path << std::endl;
        return;
    }
    output_file.write(reinterpret_cast<char*>(image_data.data()), image_data.size());
    output_file.close();
    std::cout << "Pixelated image saved to: " << output_image_path << std::endl;

    // 清理GPU内存
    CHECK_CUDA(cudaFree(calib_param_gpu));
    CHECK_CUDA(cudaFree(roi_param_gpu));
    CHECK_CUDA(cudaFree(dst_image_size_gpu));
    CHECK_CUDA(cudaFree(dst_rectangle_gpu));
    CHECK_CUDA(cudaFree(src_rectangle_gpu));

}

/*************************************** resize ***************************************/

#define INTER 11
#define INTER_SCALE (1 << INTER) // 双线性插值的缩放因子，使用整数运算来提高效率

template<typename T>
static __forceinline__ __device__ T limit(T value, T min_value, T max_value) {
    return value < min_value ? min_value : (value > max_value ? max_value : value);
}

__global__ void ResizeKernel(uint8_t* yuv_data, const int32_t* roi_param, const float* calib_param, const int32_t* dst_image_size, const bool* keep_shape, uint8_t* output_rgb_data) {
    int u = threadIdx.x + blockIdx.x * blockDim.x; // 目标图像上的x坐标
    int v = threadIdx.y + blockIdx.y * blockDim.y; // 目标图像上的y坐标
    int dst_image_width = dst_image_size[0];
    int dst_image_height = dst_image_size[1];
    if (u >= dst_image_width || v >= dst_image_height) {
        return; // 超出目标图像范围的线程不处理
    }

    int32_t src_w = roi_param[0];
    int32_t src_h = roi_param[1];
    int32_t roi_x = roi_param[2];
    int32_t roi_y = roi_param[3];
    int32_t roi_width = roi_param[4];
    int32_t roi_height = roi_param[5];

    float src_x = (u * 1.0f / dst_image_width) * roi_width + roi_x; // 目标图像上的坐标映射到ROI区域图上的坐标
    float src_y = (v * 1.0f / dst_image_height) * roi_height + roi_y; // 目标图像上的坐标映射到ROI区域图上的坐标
    int dst_index = v * dst_image_width + u; // 目标图像上的坐标对应的索引

    uchar3 rgb_data[4]; // 用于存储双线性插值需要的四个像素点的RGB值
    
    /**
     * @brief 双线性插值的步骤：
     * 1. 首先根据映射回原始图像上的坐标(src_x, src_y)
     *    计算出周围的四个像素点的坐标(x_low, y_low), (x_high, y_low), (x_low, y_high), (x_high, y_high)，
     *    其中x_low和y_low是向下取整得到的坐标，x_high和y_high是在x_low跟y_low的基础上加1得到的坐标。
     *    为了避免越界，需要对这些坐标进行限制，确保它们在原始图像的范围内。
     * 2. 计算出src_x和src_y相对于(x_low, y_low)的偏移量，分别是(src_x - x_low)和(src_y - y_low)，
     *    然后将它们乘以大整数，将偏移量从[0, 1)范围内的浮点数转换为[0, INTER_SCALE)范围内的整数，分别是lx和ly。
     *    同时计算出hx和hy，分别是INTER_SCALE减去lx和ly，表示相对于(x_high, y_high)的权重。
     * 3. 获取双线性插值需要的四个像素点的RGB值，分别是(x_low, y_low), (x_high, y_low), (x_low, y_high), (x_high, y_high)对应的像素点的RGB值。
     * 4. 根据双线性插值的公式，计算出目标图像上坐标(u, v)对应的RGB值
     */
    int x_low = floorf(src_x); // floorf函数返回小于或等于的最大整数，类似floor，cuda内置函数
    int y_low = floorf(src_y);
    int x_high = limit(x_low + 1, 0, src_w - 1);
    int y_high = limit(y_low + 1, 0, src_h - 1);
    x_low = limit(x_low, 0, src_w - 1);
    y_low = limit(y_low, 0, src_h - 1);

    int y_low_offest = rint((src_y - y_low) * INTER_SCALE); // rint函数类似round，cuda内置函数
    int x_low_offest = rint((src_x - x_low) * INTER_SCALE);
    int y_high_offest = INTER_SCALE - y_low_offest;
    int x_high_offest = INTER_SCALE - x_low_offest;

    // 获取双线性插值需要的四个像素点的RGB值，分别是(x_low, y_low), (x_high, y_low), (x_low, y_high), (x_high, y_high)对应的像素点的RGB值
    GetRGBPixel(yuv_data, x_low, y_low, src_h, src_w, rgb_data[0]);
    GetRGBPixel(yuv_data, x_high, y_low, src_h, src_w, rgb_data[1]);
    GetRGBPixel(yuv_data, x_low, y_high, src_h, src_w, rgb_data[2]);
    GetRGBPixel(yuv_data, x_high, y_high, src_h, src_w, rgb_data[3]);

    uint8_t r = 0;
    uint8_t g = 0;
    uint8_t b = 0;

    /** 
     * 横向插值：
     *  1. x_high_offest * rgb_data[0].x + x_low_offest * rgb_data[1].x
     *     表示在y_low这一行上，src_x相对于(x_low, y_low)的权重分别是x_high_offest和x_low_offest，
     *     乘以对应像素点的R分量值，得到在y_low这一行上src_x对应的R分量值。
     *  2. x_high_offest * rgb_data[2].x + x_low_offest * rgb_data[3].x
     *     表示在y_high这一行上，src_x相对于(x_low, y_high)的权重分别是x_high_offest和x_low_offest，
     *     乘以对应像素点的R分量值，得到在y_high这一行上src_x对应的R分量值。
     * 再纵向插值：
     *  1. y_high_offest * (在y_low这一行上src_x对应的R分量值)
     *  2. y_low_offest * (在y_high这一行上src_x对应的R分量值)
     * 最后将两部分的结果相加，得到最终的R分量值。G和B分量的计算公式与R分量类似，只需要将R替换为G或B即可。
     *
     * 下面的计算公式中：
     * 1. >> 4：x_high_offest和x_low_offest是乘以了INTER_SCALE(1 << INTER)的整数，单个像素值是255(1<<8)，
     *          这里先简单降低整数的位数，避免后续计算中整数溢出，降低计算的复杂度。
     * 2. >> 16：y_high_offest和y_low_offest也是乘以了INTER_SCALE(1 << INTER)的整数，
     *          前面已经降低了x_high_offest和x_low_offest的位数，这里再降低y_high_offest和y_low_offest的位数，避免整数溢出，降低计算的复杂度。
     * 3. + 2) >> 2：等价于 + (1<1) >> 2，即在除以4之前先加上2，起到四舍五入的作用，进一步提高计算的精度。
     * 4. 整体上左移动了INTER_SCALE*2的位数(y_high_offest和y_low_offest都是左移INTER_SCALE)，
     *    所以需要右移INTER_SCALE*2的位数来恢复到正常的范围内，避免整数溢出，同时保持计算的精度。
     */
    r = (((y_high_offest * ((x_high_offest * rgb_data[0].x + x_low_offest * rgb_data[1].x) >> 4)) >> 16)
        + ((y_low_offest * ((x_high_offest * rgb_data[2].x + x_low_offest * rgb_data[3].x) >> 4)) >> 16) + 2) >> 2;
    g = (((y_high_offest * ((x_high_offest * rgb_data[0].y + x_low_offest * rgb_data[1].y) >> 4)) >> 16)
        + ((y_low_offest * ((x_high_offest * rgb_data[2].y + x_low_offest * rgb_data[3].y) >> 4)) >> 16) + 2) >> 2;
    b = (((y_high_offest * ((x_high_offest * rgb_data[0].z + x_low_offest * rgb_data[1].z) >> 4)) >> 16)
        + ((y_low_offest * ((x_high_offest * rgb_data[2].z + x_low_offest * rgb_data[3].z) >> 4)) >> 16) + 2) >> 2;

    output_rgb_data[dst_index] = r;
    output_rgb_data[dst_image_width * dst_image_height + dst_index] = g;
    output_rgb_data[2 * dst_image_width * dst_image_height + dst_index] = b;
}

void resize() {
    // 输入图像数据，假设是yuv格式
    std::string input_image_path = "/mnt/workspace/cgz_workspace/Exercise/camera_example/input/frontwide_3840_2048_nv12.yuv";
    int image_width = 3840;
    int image_height = 2048;
    int yuv_size = image_width * image_height * 3 / 2; // nv12格式的图像大小
    std::vector<uint8_t> yuv_data(yuv_size);
    std::ifstream input_file(input_image_path, std::ios::binary|std::ios::ate);
    if (!input_file.is_open()) {
        std::cerr << "Failed to open input image file: " << input_image_path << std::endl;
        return;
    }
    std::size_t file_size = input_file.tellg();
    if (file_size != yuv_size) {
        std::cerr << "Input image file size does not match expected size: " << file_size << " vs " << yuv_size << std::endl;
        return;
    }
    input_file.seekg(0, std::ios::beg);
    input_file.read(reinterpret_cast<char*>(yuv_data.data()), yuv_size);
    input_file.close();
    std::cout << "Input image size: " << file_size << " bytes" << std::endl;

    // 设置输入图像数据的GPU内存
    uint8_t* yuv_data_gpu = nullptr;
    CHECK_CUDA(cudaMalloc(&yuv_data_gpu, yuv_size));
    CHECK_CUDA(cudaMemcpy(yuv_data_gpu, yuv_data.data(), yuv_size, cudaMemcpyHostToDevice));

    // 设置roi
    int32_t roi_param[6] = {image_width, image_height, 0, 0, image_width, image_height}; // roi参数，假设是全图
    int32_t* roi_param_gpu = nullptr;
    CHECK_CUDA(cudaMalloc(&roi_param_gpu, sizeof(roi_param)));
    CHECK_CUDA(cudaMemcpy(roi_param_gpu, roi_param, sizeof(roi_param), cudaMemcpyHostToDevice));
    std::cout << "ROI: image_width=" << roi_param[0] << ", image_height=" << roi_param[1] << std::endl;
    std::cout << "ROI: x=" << roi_param[2] << ", y=" << roi_param[3] << ", width=" << roi_param[4] << ", height=" << roi_param[5] << std::endl;

    // 设置输出图像的尺寸
    int32_t dst_image_size[2] = {image_width / 2, image_height / 2}; // 输出图像的尺寸
    int32_t* dst_image_size_gpu = nullptr;
    CHECK_CUDA(cudaMalloc(&dst_image_size_gpu, sizeof(dst_image_size)));
    CHECK_CUDA(cudaMemcpy(dst_image_size_gpu, dst_image_size, sizeof(dst_image_size), cudaMemcpyHostToDevice));
    std::cout << "Output image size: width=" << dst_image_size[0] << ", height=" << dst_image_size[1] << std::endl;

    // 设置输出图像数据的GPU内存
    uint8_t* output_rgb_data_cpu = nullptr; // 输出图像数据的CPU内存
    output_rgb_data_cpu = new uint8_t[dst_image_size[0] * dst_image_size[1] * 3]; // 输出图像数据的大小，假设是RGB格式
    memset(output_rgb_data_cpu, 0, dst_image_size[0] * dst_image_size[1] * 3); // 初始化输出图像数据
    uint8_t* output_rgb_data_gpu = nullptr; // 输出图像数据的GPU内存
    CHECK_CUDA(cudaMalloc(&output_rgb_data_gpu, dst_image_size[0] * dst_image_size[1] * 3));
    CHECK_CUDA(cudaMemcpy(output_rgb_data_gpu, output_rgb_data_cpu, dst_image_size[0] * dst_image_size[1] * 3, cudaMemcpyHostToDevice));

    // 设置相机内参和畸变参数
    float calib_param[12] = {1906.6, 1906.18, 1923.26, 1022.45, -0.0299548, -0.00364585, -0.00155829, 0.00104736}; // fx, fy, cx, cy, k1, k2, k3, k4
    float fx_scale = 1.0;
    float fy_scale = 1.0;
    float cx_offset = 0.0;
    float cy_offset = 0.0;
    calib_param[8] = calib_param[0] / fx_scale;
    calib_param[9] = calib_param[1] / fy_scale;
    calib_param[10] = cx_offset;
    calib_param[11] = cy_offset;
    float* calib_param_gpu = nullptr;
    CHECK_CUDA(cudaMalloc(&calib_param_gpu, sizeof(calib_param)));
    CHECK_CUDA(cudaMemcpy(calib_param_gpu, calib_param, sizeof(calib_param), cudaMemcpyHostToDevice));
    std::cout << "Camera intrinsics: fx=" << calib_param[0] << ", fy=" << calib_param[1] << ", cx=" << calib_param[2] << ", cy=" << calib_param[3] << std::endl;
    std::cout << "Distortion coefficients: k1=" << calib_param[4] << ", k2=" << calib_param[5] << ", k3=" << calib_param[6] << ", k4=" << calib_param[7] << std::endl;
    std::cout << "fx_scale=" << fx_scale << ", fy_scale=" << fy_scale << ", cx_offset=" << cx_offset << ", cy_offset=" << cy_offset << std::endl;
    std::cout << "fx_rd=" << calib_param[8] << ", fy_rd=" << calib_param[9] << ", cx_offset=" << calib_param[10] << ", cy_offset=" << calib_param[11] << std::endl;

    // 设置是否保持图像内容的宽高比
    bool keep_shape = false;
    bool* keep_shape_gpu = nullptr;
    CHECK_CUDA(cudaMalloc(&keep_shape_gpu, sizeof(bool)));
    CHECK_CUDA(cudaMemcpy(keep_shape_gpu, &keep_shape, sizeof(bool), cudaMemcpyHostToDevice));
    std::cout << "Keep shape: " << (keep_shape ? "true" : "false") << std::endl;

    // 启动kernel进行图像缩放和格式转换
    dim3 block_dim(64, 4); // 每个block有64*4=256个线程
    dim3 grid_dim((dst_image_size[0] + block_dim.x - 1) / block_dim.x, (dst_image_size[1] + block_dim.y - 1) / block_dim.y); // 根据输出图像的尺寸计算需要多少个block
    std::cout << "Block dim: (" << block_dim.x << ", " << block_dim.y << ", " << block_dim.z << ")" << std::endl;
    std::cout << "Grid dim: (" << grid_dim.x << ", " << grid_dim.y << ", " << grid_dim.z << ")" << std::endl;
    ResizeKernel<<<grid_dim, block_dim>>>(yuv_data_gpu, roi_param_gpu, calib_param_gpu, dst_image_size_gpu, keep_shape_gpu, output_rgb_data_gpu);
    CHECK_CUDA(cudaDeviceSynchronize()); // 等待kernel执行完成

    // 将处理后的图像数据从GPU内存复制回CPU内存
    CHECK_CUDA(cudaMemcpy(output_rgb_data_cpu, output_rgb_data_gpu, dst_image_size[0] * dst_image_size[1] * 3, cudaMemcpyDeviceToHost));
    std::string output_image_path = "/mnt/workspace/cgz_workspace/Exercise/camera_example/output/undistort_frontwide_1920_1024_resize.rgb"; // 输出图像的路径，假设是bgr格式
    std::ofstream output_file(output_image_path, std::ios::binary);
    if (!output_file.is_open()) {
        std::cerr << "Failed to open output image file: " << output_image_path << std::endl;
        return;
    }
    output_file.write(reinterpret_cast<char*>(output_rgb_data_cpu), dst_image_size[0] * dst_image_size[1] * 3);
    output_file.close();
    std::cout << "Output image saved to: " << output_image_path << std::endl;

    // 保存成nv12格式，方便后续验证
    std::string output_nv12_image_path = "/mnt/workspace/cgz_workspace/Exercise/camera_example/output/undistort_frontwide_1920_1024_nv12_resize.yuv";
    std::ofstream output_nv12_file(output_nv12_image_path, std::ios::binary);
    if (!output_nv12_file.is_open()) {
        std::cerr << "Failed to open output nv12 image file: " << output_nv12_image_path << std::endl;
        return;
    }
    // 将RGB格式的图像数据转换为NV12格式并保存
    std::vector<uint8_t> nv12_data(dst_image_size[0] * dst_image_size[1] * 3 / 2); // NV12格式的图像大小
    for (int v = 0; v < dst_image_size[1]; ++v) {
        for (int u = 0; u < dst_image_size[0]; ++u) {
            int dst_index = v * dst_image_size[0] + u;
            uint8_t R = output_rgb_data_cpu[dst_index]; // R分量
            uint8_t G = output_rgb_data_cpu[dst_image_size[0] * dst_image_size[1] + dst_index]; // G分量
            uint8_t B = output_rgb_data_cpu[2 * dst_image_size[0] * dst_image_size[1] + dst_index]; // B分量
            // RGB转YUV的计算公式(ITU-R BT.601标准)：
            // Y = 0.299 * R + 0.587 * G + 0.114 * B
            // U = -0.169 * R - 0.331 * G + 0.5 * B + 128
            // V = 0.5 * R - 0.419 * G - 0.081 * B + 128
            uint8_t Y = static_cast<uint8_t>(0.299 * R + 0.587 * G + 0.114 * B);
            uint8_t U = static_cast<uint8_t>(-0.169 * R - 0.331 * G + 0.5 * B + 128);
            uint8_t V = static_cast<uint8_t>(0.5 * R - 0.419 * G - 0.081 * B + 128);
            nv12_data[v * dst_image_size[0] + u] = Y; // Y分量
            if (v % 2 == 0 && u % 2 == 0) {
                int uv_index = (dst_image_size[0] * dst_image_size[1]) + (v / 2) * dst_image_size[0] + (u / 2) * 2; // UV分量的索引，假设是NV12格式，UV分量交错存储
                nv12_data[uv_index + 0] = U; // U分量
                nv12_data[uv_index + 1] = V; // V分量
            }
        }
    }
    output_nv12_file.write(reinterpret_cast<char*>(nv12_data.data()), nv12_data.size());
    output_nv12_file.close();
    std::cout << "Output NV12 image saved to: " << output_nv12_image_path << std::endl;
}

int main() {
    std::cout << "=================== undistort ===================" << std::endl;
    undistort();
    std::cout << "=================== pixlate ===================" << std::endl;
    pixlate();
    std::cout << "=================== resize ===================" << std::endl;
    resize();
    return 0;
}
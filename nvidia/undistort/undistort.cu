#include <bits/c++config.h>
#include <bits/stdint-intn.h>
#include <bits/stdint-uintn.h>
#include <iostream>
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

__device__ void __forceinline__ GetRGBPixel(uint8_t* yuv_data, int32_t x, int32_t y, int32_t height, int32_t pitch, uchar3& rgb_pixel) {
    int y_index = y * pitch + x; // Y分量的索引
    int u_index = height * pitch + (y / 2) * (pitch / 2) + (x / 2); // U分量的索引
    int v_index = height * pitch + (height / 2) * (pitch / 2) + (y / 2) * (pitch / 2) + (x / 2); // V分量的索引

    uint8_t Y = yuv_data[y_index];
    uint8_t U = yuv_data[u_index];
    uint8_t V = yuv_data[v_index];

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
    std::string input_image_path = "/mnt/workspace/cgz_workspace/Exercise/camera_example/input/frontwide_3840_2048_420p.yuv";
    int image_width = 3840;
    int image_height = 2048;
    int yuv_size = image_width * image_height * 3 / 2; // yuv420p格式的图像大小
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

    // 设置输入输出图像数据的GPU内存
    uint8_t* yuv_data_gpu = nullptr;
    CHECK_CUDA(cudaMalloc(&yuv_data_gpu, yuv_size));
    CHECK_CUDA(cudaMemcpy(yuv_data_gpu, yuv_data.data(), yuv_size, cudaMemcpyHostToDevice));
    uint8_t* output_rgb_data_cpu = nullptr; // 输出图像数据的CPU内存
    output_rgb_data_cpu = new uint8_t[image_width * image_height * 3]; // 输出图像数据的大小，假设是RGB格式
    memset(output_rgb_data_cpu, 0, image_width * image_height * 3); // 初始化输出图像数据
    uint8_t* output_rgb_data_gpu = nullptr; // 输出图像数据的GPU内存
    CHECK_CUDA(cudaMalloc(&output_rgb_data_gpu, image_width * image_height * 3));
    CHECK_CUDA(cudaMemcpy(output_rgb_data_gpu, output_rgb_data_cpu, image_width * image_height * 3, cudaMemcpyHostToDevice));

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
    CHECK_CUDA(cudaMemcpy(output_rgb_data_cpu, output_rgb_data_gpu, image_width * image_height * 3, cudaMemcpyDeviceToHost));
    std::string output_image_path = "/mnt/workspace/cgz_workspace/Exercise/camera_example/output/frontwide_1920_1024_3.bgr"; // 输出图像的路径，假设是bgr格式
    std::ofstream output_file(output_image_path, std::ios::binary);
    if (!output_file.is_open()) {
        std::cerr << "Failed to open output image file: " << output_image_path << std::endl;
        return;
    }
    output_file.write(reinterpret_cast<char*>(output_rgb_data_cpu), image_width * image_height * 3);
    output_file.close();
    std::cout << "Output image saved to: " << output_image_path << std::endl;

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

int main() {
    std::cout << "=================== undistort ===================" << std::endl;
    undistort();
    return 0;
}
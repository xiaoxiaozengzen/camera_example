#include <iostream>
#include <vector>
#include <fstream>

#include <cuda_runtime.h>
#include <npp.h>
#include <nppi.h>

#define CHECK_CUDA(call) do { cudaError_t e = (call); if (e != cudaSuccess) { std::cerr << "CUDA error: " << cudaGetErrorString(e) << std::endl; exit(1); } } while(0)
#define CHECK_NPP(call) do { NppStatus s = (call); if (s != NPP_SUCCESS) { std::cerr << "NPP error: " << s << std::endl; exit(1); } } while(0)

/**
 * enum NppStatus {
 *     NPP_NO_ERROR = 0,               // 无错误
 *     NPP_SUCCESS = NPP_NO_ERROR, // 成功
 *     ...
 * };
 *
 * struct NppiSize {
 *     int width;  // 图像宽度
 *     int height; // 图像高度
 * };
 *
 * struct NppiRect {
 *     int x;      // 矩形左上角的X坐标
 *     int y;      // 矩形左上角的Y坐标
 *     int width;  // 矩形的宽度
 *     int height; // 矩形的高度
 * };
 * 
 * struct NppiPoint {
 *     int x; // X坐标
 *     int y; // Y坐标
 * };
 *
 * struct NppStreamContext {
 *     cudaStream_t hStream;  // 绑定的CUDA流
 *     int nCudaDevice;       // From cudaGetDevice()
 *     int nMultiProcessorCount; // From cudaGetDeviceProperties()
 *     int nMaxThreadsPerBlock; // From cudaGetDeviceProperties()
 *     size_t nSharedMemPerBlock; // From cudaGetDeviceProperties()
 *     int nCudaDevAttrComputeCapabilityMajor; // From cudaGetDeviceProperties()
 *     int nCudaDevAttrComputeCapabilityMinor; // From cudaGetDeviceProperties()
 *     unsigned int nStreamFlags; // From cudaStreamGetFlags()
 *     int nReserve0; // 保留字段
 * };
 */

void npp_sample_example() {
  // 创建stream，主要给带ctx参数的NPP函数使用，普通的NPP函数会在当前设置的流上执行
  cudaStream_t stream = nullptr;
  unsigned int flags;
  CHECK_CUDA(cudaStreamGetFlags(stream, &flags));
  if(flags == cudaStreamNonBlocking) {
    std::cout << "Stream flags is non-blocking" << std::endl;
  } else if (flags == cudaStreamDefault) {
    std::cout << "Stream flags is default" << std::endl;
  } else {
    std::cout << "Stream flags has unknown flags: " << flags << std::endl;
  }
  int priorityHigh, priorityLow;
  cudaDeviceGetStreamPriorityRange(&priorityLow, &priorityHigh);
  cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, priorityHigh);

  // 查看当前Npp所属的CUDA流
  cudaStream_t npp_get_stream = nppGetStream();
  if(npp_get_stream) {
    std::cout << "Current NPP CUDA Stream: " << npp_get_stream << std::endl;
  } else {
    std::cout << "Current NPP CUDA Stream: Default Stream (0)" << std::endl;
    CHECK_CUDA(cudaStreamGetFlags(npp_get_stream, &flags));
    if(flags == cudaStreamNonBlocking) {
      std::cout << "npp_get_stream flags is non-blocking" << std::endl;
    } else if (flags == cudaStreamDefault) {
      std::cout << "npp_get_stream flags is default" << std::endl;
    } else {
      std::cout << "npp_get_stream flags has unknown flags: " << flags << std::endl;
    }
  }

  /**
   * @brief 设置NPP使用指定的CUDA流，之后调用的NPP函数会在这个流上执行
   * @note 可以多次调用nppSetStream切换不同的CUDA流，NPP函数会在当前设置的流上执行
   */
  nppSetStream(stream);
  npp_get_stream = nppGetStream();
  if(npp_get_stream) {
    std::cout << "NPP CUDA Stream after setting: " << npp_get_stream << std::endl;
  } else {
    std::cout << "NPP CUDA Stream after setting: Default Stream (0)" << std::endl;
  }
  
  // 设置会默认的CUDA流，等同于nppSetStream(0)
  nppSetStream(0);
  npp_get_stream = nppGetStream();
  if(npp_get_stream) {
    std::cout << "NPP CUDA Stream after resetting to default: " << npp_get_stream << std::endl;
  } else {
    std::cout << "NPP CUDA Stream after resetting to default: Default Stream (0)" << std::endl;
  }

  // 获取当前默认的NPP管理的CUDA流上下文信息
  NppStreamContext nppStreamCtx;
  /**
   * @brief 获取当前NPP管理的cudastram context信息，即调用了nppSetStream后所设置的cudastream
   */
  CHECK_NPP(nppGetStreamContext(&nppStreamCtx));
  std::cout << "NPP Stream Context:" << std::endl;
  if(nppStreamCtx.hStream) {
    std::cout << "  CUDA Stream: " << nppStreamCtx.hStream << std::endl;
  } else {
    std::cout << "  CUDA Stream: Default Stream (0)" << std::endl;
  }
  std::cout << "  CUDA Device: " << nppStreamCtx.nCudaDeviceId << std::endl;
  std::cout << "  MultiProcessor Count: " << nppStreamCtx.nMultiProcessorCount << std::endl;
  std::cout << "  Max Threads Per Block: " << nppStreamCtx.nMaxThreadsPerBlock << std::endl;
  std::cout << "  Shared Memory Per Block: " << nppStreamCtx.nSharedMemPerBlock << " bytes" << std::endl;
  std::cout << "  Compute Capability: " << nppStreamCtx.nCudaDevAttrComputeCapabilityMajor << "." << nppStreamCtx.nCudaDevAttrComputeCapabilityMinor << std::endl;
  std::cout << "  Stream Flags: " << nppStreamCtx.nStreamFlags << std::endl;

  // 创建新的nppStreamContext并设置一个新的CUDA流
  NppStreamContext newNppStreamCtx;
  CHECK_NPP(nppGetStreamContext(&newNppStreamCtx));
  CHECK_CUDA(cudaGetDevice(&newNppStreamCtx.nCudaDeviceId));
  struct cudaDeviceProp devProp;
  cudaGetDeviceProperties(&devProp, newNppStreamCtx.nCudaDeviceId);
  newNppStreamCtx.hStream = stream;
  newNppStreamCtx.nMultiProcessorCount = devProp.multiProcessorCount;
  newNppStreamCtx.nMaxThreadsPerBlock = devProp.maxThreadsPerBlock;
  newNppStreamCtx.nSharedMemPerBlock = devProp.sharedMemPerBlock;
  newNppStreamCtx.nCudaDevAttrComputeCapabilityMajor = devProp.major;
  newNppStreamCtx.nCudaDevAttrComputeCapabilityMinor = devProp.minor;
  CHECK_CUDA(cudaStreamGetFlags(stream, &newNppStreamCtx.nStreamFlags));
  std::cout << "New NPP Stream Context:" << std::endl;
  if(newNppStreamCtx.hStream) {
    std::cout << "  CUDA Stream: " << newNppStreamCtx.hStream << std::endl;
  } else {
    std::cout << "  CUDA Stream: Default Stream (0)" << std::endl;
  }
  std::cout << "  CUDA Device: " << newNppStreamCtx.nCudaDeviceId << std::endl;
  std::cout << "  MultiProcessor Count: " << newNppStreamCtx.nMultiProcessorCount << std::endl;
  std::cout << "  Max Threads Per Block: " << newNppStreamCtx.nMaxThreadsPerBlock << std::endl;
  std::cout << "  Shared Memory Per Block: " << newNppStreamCtx.nSharedMemPerBlock << " bytes" << std::endl;
  std::cout << "  Compute Capability: " << newNppStreamCtx.nCudaDevAttrComputeCapabilityMajor << "." << newNppStreamCtx.nCudaDevAttrComputeCapabilityMinor << std::endl;
  std::cout << "  Stream Flags: " << newNppStreamCtx.nStreamFlags << std::endl;


  // 生成一个简单的RGB图像
  const int w = 8, h = 6;
  const int channels = 3;
  const int srcStep = w * channels; // bytes per row (interleaved 8u)
  const int dstStep = w * channels;
  std::vector<unsigned char> h_src(w * h * channels);
  std::vector<unsigned char> h_dst(w * h * channels, 0);

  // 填充源图像数据
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      int idx = (y * w + x) * channels;
      h_src[idx + 0] = (unsigned char)(x * 10); // R
      h_src[idx + 1] = (unsigned char)(y * 20); // G
      h_src[idx + 2] = (unsigned char)( (x+y) * 5 ); // B
    }

  unsigned char *d_src=nullptr;
  unsigned char *d_dst=nullptr;
  unsigned char *d_dst_with_ctx=nullptr;
  CHECK_CUDA(cudaMalloc(&d_src, h_src.size()));
  CHECK_CUDA(cudaMalloc(&d_dst, h_dst.size()));
  CHECK_CUDA(cudaMalloc(&d_dst_with_ctx, h_dst.size()));
  CHECK_CUDA(cudaMemcpy(d_src, h_src.data(), h_src.size(), cudaMemcpyHostToDevice));

  NppiSize roi = { w, h };

  /**
   * @brief 给一个8U3C的RGB图像，源图按照ROI区域和给定源图步长，拷贝到目标图像的对应区域，目标图按照给定的目标图步长存储
   * @param pSrc 源图像数据指针
   * @param nSrcStep 源图像每行的字节数（stride）
   * @param pDst 目标图像数据指针
   * @param nDstStep 目标图像每行的字节数（stride）
   * @param oSizeROI 图像的宽度
   * @return NppStatus，表示函数执行的状态
   *
   * @note 函数名字简单解析：
   *       nppi：NPP的图像处理模块
   *       Copy：拷贝
   *       8u：像素的数据类型是8-bit unsigned
   *       C3：三通道交错存储，例如RGBRGB
   *       R：带ROI的版本
   */
  CHECK_NPP(nppiCopy_8u_C3R(d_src, srcStep, d_dst, dstStep, roi));

  // 使用带ctx参数的版本在指定的CUDA流上执行拷贝操作
  CHECK_NPP(nppiCopy_8u_C3R_Ctx(d_src, srcStep, d_dst_with_ctx, dstStep, roi, newNppStreamCtx));

  // 将数据从device拷贝回host并打印部分结果
  CHECK_CUDA(cudaMemcpy(h_dst.data(), d_dst, h_dst.size(), cudaMemcpyDeviceToHost));
  std::cout << "After nppiCopy, pixel (0,1): R G B = "
            << int(h_dst[3]) << " " << int(h_dst[4]) << " " << int(h_dst[5]) << std::endl;

  CHECK_CUDA(cudaMemcpy(h_dst.data(), d_dst_with_ctx, h_dst.size(), cudaMemcpyDeviceToHost));
  std::cout << "After nppiCopy with ctx, pixel (0,1): R G B = "
            << int(h_dst[3]) << " " << int(h_dst[4]) << " " << int(h_dst[5]) << std::endl;

  // 释放device端内存和CUDA流
  cudaFree(d_src);
  cudaFree(d_dst);
  cudaFree(d_dst_with_ctx);
  cudaStreamDestroy(stream);
}

void npp_resize_example() {
  // 输入输出图像路径
  std::string input_image_path = "/mnt/workspace/cgz_workspace/Exercise/camera_example/input/650_yuv420p.yuv";
  std::string output_image_path = "/mnt/workspace/cgz_workspace/Exercise/camera_example/output/650_npp_resized.yuv";
  
  // 获取输入图像数据
  std::ifstream input_file(input_image_path, std::ios::binary | std::ios::ate);
  if (!input_file.is_open()) {
    std::cerr << "Failed to open input file: " << input_image_path << std::endl;
    return;
  }
  std::streamsize input_size = input_file.tellg();
  input_file.seekg(0, std::ios::beg);
  std::vector<unsigned char> h_src(input_size);
  if (!input_file.read(reinterpret_cast<char*>(h_src.data()), input_size)) {
    std::cerr << "Failed to read input file: " << input_image_path << std::endl;
    return;
  }
  std::cout << "Read input image: " << input_image_path << ", size: " << input_size << " bytes" << std::endl;

  // 定义源图像和目标图像的尺寸
  const int src_width = 1080;
  const int src_height = 1920;
  const int dst_width = 540;
  const int dst_height = 540;
  std::cout << "Source image size: " << src_width << "x" << src_height << std::endl;
  std::cout << "Destination image size: " << dst_width << "x" << dst_height << std::endl;

  // 计算YUV420P格式中Y、U、V分量的大小和帧大小
  const size_t src_y_size = src_width * src_height;
  const size_t src_u_size = (src_width / 2) * (src_height / 2);
  const size_t src_v_size = src_u_size;
  const size_t src_frame_size = src_y_size + src_u_size + src_v_size;

  // host端源图像和目标图像数据指针
  const unsigned char* h_src_y = h_src.data();
  const unsigned char* h_src_u = h_src.data() + src_y_size;
  const unsigned char* h_src_v = h_src.data() + src_y_size + src_u_size;
  const size_t rgb_channels = 3;
  const size_t src_rgb_size = src_width * src_height * rgb_channels;
  std::vector<Npp8u> h_src_rgb(src_rgb_size);
  const size_t dst_rgb_size = dst_width * dst_height * rgb_channels;
  std::vector<Npp8u> h_dst_rgb(dst_rgb_size);

  // 分配device端内存并将数据从host端拷贝到device端
  Npp8u *d_src_y=nullptr, *d_src_u=nullptr, *d_src_v=nullptr;
  Npp8u *d_src_rgb=nullptr;
  Npp8u *d_dst_rgb=nullptr;
  CHECK_CUDA(cudaMalloc(&d_src_y, src_y_size));
  CHECK_CUDA(cudaMalloc(&d_src_u, src_u_size));
  CHECK_CUDA(cudaMalloc(&d_src_v, src_v_size));
  CHECK_CUDA(cudaMalloc(&d_src_rgb, src_rgb_size));
  CHECK_CUDA(cudaMalloc(&d_dst_rgb, dst_rgb_size));
  CHECK_CUDA(cudaMemcpy(d_src_y, h_src_y, src_y_size, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_src_u, h_src_u, src_u_size, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_src_v, h_src_v, src_v_size, cudaMemcpyHostToDevice));

  Npp8u* yuv_data[3] = { d_src_y, d_src_u, d_src_v };
  int yuv_step[3] = { src_width, src_width / 2, src_width / 2 };

  /**
   * @brief 3通道uint8_t的YUV420P图像转换为3通道uint8_t的RGB图像
   * @param pSrc 一个数组，指向输入图像的Y、U、V分量数据指针，数组长度为3
   * @param rSrcStep 一个数组，指向输入图像的Y、U、V分量每行的字节数(stride)，数组长度为3
   * @param pDst 指向输出图像数据的指针，输出图像是3通道RGB格式
   * @param nDstStep 输出图像每行的字节数(stride)
   * @param oSizeROI roi区域的尺寸，表示输入图像和输出图像的宽高
   */
  CHECK_NPP(nppiYUV420ToRGB_8u_P3C3R(yuv_data, yuv_step, d_src_rgb, src_width * rgb_channels, { src_width, src_height }));
  CHECK_CUDA(cudaMemcpy(h_src_rgb.data(), d_src_rgb, src_rgb_size, cudaMemcpyDeviceToHost));

  NppiSize src_size = { src_width, src_height };
  NppiRect src_roi = { 0, 0, src_width, src_height };
  int src_step = src_width * rgb_channels; // RGB图像每行的字节数
  NppiSize dst_size = { dst_width, dst_height };
  NppiRect dst_roi = { 0, 0, dst_width, dst_height };
  int dst_step = dst_width * rgb_channels; // RGB图像每行的字节数
  /**
   * @brief resize image
   * @param pSrc 源图像数据指针
   * @param nSrcStep 源图像每行的字节数(stride)
   * @param oSrcSize 源图像的尺寸
   * @param oSrcROI 源图像的ROI区域
   * @param pDst 目标图像数据指针
   * @param nDstStep 目标图像每行的字节数(stride)
   * @param oDstSize 目标图像的尺寸
   * @param oDstROI 目标图像的ROI区域
   * @param eInterpolationType 插值算法类型
   */
  CHECK_NPP(nppiResize_8u_C3R(d_src_rgb, src_step, src_size, src_roi, d_dst_rgb, dst_step, dst_size, dst_roi, NPPI_INTER_LINEAR));

  // 将数据从device端拷贝回host端并写入输出文件
  CHECK_CUDA(cudaMemcpy(h_dst_rgb.data(), d_dst_rgb, dst_rgb_size, cudaMemcpyDeviceToHost));

  // RGB转YUV420P
  std::vector<unsigned char> h_dst_yuv420p(dst_height * dst_width * 3 / 2);
  for(int y = 0; y < dst_height; y++) {
    for(int x = 0; x < dst_width; x++) {
      int idx_rgb = (y * dst_width + x) * rgb_channels;
      unsigned char r = h_dst_rgb[idx_rgb + 0];
      unsigned char g = h_dst_rgb[idx_rgb + 1];
      unsigned char b = h_dst_rgb[idx_rgb + 2];
      unsigned char y_val = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
      unsigned char u_val = static_cast<unsigned char>(-0.169f * r - 0.331f * g + 0.5f * b + 128);
      unsigned char v_val = static_cast<unsigned char>(0.5f * r - 0.419f * g - 0.081f * b + 128);
      h_dst_yuv420p[y * dst_width + x] = y_val;
      if(y % 2 == 0 && x % 2 == 0) {
        int idx_uv = (y / 2) * (dst_width / 2) + (x / 2);
        h_dst_yuv420p[dst_height * dst_width + idx_uv] = u_val;
        h_dst_yuv420p[dst_height * dst_width + (dst_width / 2) * (dst_height / 2) + idx_uv] = v_val;
      }
    }
  }

  std::ofstream output_file(output_image_path, std::ios::binary);
  if (!output_file.is_open()) {
    std::cerr << "Failed to open output file: " << output_image_path << std::endl;
    return;
  }
  output_file.write(reinterpret_cast<char*>(h_dst_yuv420p.data()), h_dst_yuv420p.size());
  std::cout << "Wrote output image: " << output_image_path << ", size: " << h_dst_yuv420p.size() << " bytes" << std::endl;

  // 释放device端内存
  CHECK_CUDA(cudaFree(d_src_y));
  CHECK_CUDA(cudaFree(d_src_u));
  CHECK_CUDA(cudaFree(d_src_v));
  CHECK_CUDA(cudaFree(d_src_rgb));
  CHECK_CUDA(cudaFree(d_dst_rgb));
}

void npp_crop_example() {
  // 输入输出图像路径
  std::string input_image_path = "/mnt/workspace/cgz_workspace/Exercise/camera_example/input/650_yuv420p.yuv";
  std::string output_image_path = "/mnt/workspace/cgz_workspace/Exercise/camera_example/output/650_npp_crop.yuv";
  
  // 获取输入图像数据
  std::ifstream input_file(input_image_path, std::ios::binary);
  if (!input_file.is_open()) {
    std::cerr << "Failed to open input file: " << input_image_path << std::endl;
    return;
  }
  std::vector<unsigned char> h_src(std::istreambuf_iterator<char>(input_file), {});
  input_file.close();
  std::cout << "Read input image: " << input_image_path << ", size: " << h_src.size() << " bytes" << std::endl;

  // 定义源图像的尺寸和裁剪区域
  const int src_width = 1080;
  const int src_height = 1920;
  const int crop_x = 270;
  const int crop_y = 480;
  const int crop_width = 540;
  const int crop_height = 960;
  if(crop_x + crop_width > src_width || crop_y + crop_height > src_height) {
    std::cerr << "Crop region exceeds source image bounds!" << std::endl;
    return;
  }

  // 计算YUV420P格式中Y、U、V分量的大小和帧大小
  const size_t src_y_size = src_width * src_height;
  const size_t src_u_size = (src_width / 2) * (src_height / 2);
  const size_t src_v_size = src_u_size;
  const size_t src_frame_size = src_y_size + src_u_size + src_v_size;
  if(h_src.size() < src_frame_size) {
    std::cerr << "Input file size is smaller than expected for YUV420P frame!" << std::endl;
    return;
  }
  const int dst_width = crop_width;
  const int dst_height = crop_height;
  const size_t dst_y_size = dst_width * dst_height;
  const size_t dst_u_size = (dst_width / 2) * (dst_height / 2);
  const size_t dst_v_size = dst_u_size;
  const size_t dst_frame_size = dst_y_size + dst_u_size + dst_v_size;
  std::vector<unsigned char> h_dst(dst_frame_size, 128);

  // host端源图像和目标图像数据指针
  const unsigned char* h_src_y = h_src.data();
  const unsigned char* h_src_u = h_src.data() + src_y_size;
  const unsigned char* h_src_v = h_src.data() + src_y_size + src_u_size;
  unsigned char* h_dst_y = h_dst.data();
  unsigned char* h_dst_u = h_dst.data() + dst_y_size;
  unsigned char* h_dst_v = h_dst.data() + dst_y_size + dst_u_size;

  // 获取当前NPP CUDA流和流上下文信息
  CHECK_NPP(nppSetStream(0)); // 使用默认流

  // 分配device端内存并将数据从host端拷贝到device端
  unsigned char *d_src_y=nullptr, *d_src_u=nullptr, *d_src_v=nullptr;
  unsigned char *d_dst_y=nullptr, *d_dst_u=nullptr, *d_dst_v=nullptr;
  CHECK_CUDA(cudaMalloc(&d_src_y, src_y_size));
  CHECK_CUDA(cudaMalloc(&d_src_u, src_u_size));
  CHECK_CUDA(cudaMalloc(&d_src_v, src_v_size));
  CHECK_CUDA(cudaMalloc(&d_dst_y, dst_y_size));
  CHECK_CUDA(cudaMalloc(&d_dst_u, dst_u_size));
  CHECK_CUDA(cudaMalloc(&d_dst_v, dst_v_size));
  CHECK_CUDA(cudaMemcpy(d_src_y, h_src_y, src_y_size, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_src_u, h_src_u, src_u_size, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_src_v, h_src_v, src_v_size, cudaMemcpyHostToDevice));

  std::cout << "Start cropping Y component..." << std::endl;
  // Y crop
  // 用偏移后的源指针+ROI尺寸的方式实现裁剪
  const int srcY_step = src_width; // Y分量每行的字节数
  const int dstY_step = dst_width; // Y分量每行的字节数
  NppiSize y_roi_size = { crop_width, crop_height };
  const Npp8u* d_src_y_crop = d_src_y + crop_y * srcY_step + crop_x; // 计算裁剪区域的起始地址
  CHECK_NPP(nppiCopy_8u_C1R(d_src_y_crop, srcY_step, d_dst_y, dstY_step, y_roi_size));

  // 将数据从device端拷贝回host端并写入输出文件
  CHECK_CUDA(cudaMemcpy(h_dst_y, d_dst_y, dst_y_size, cudaMemcpyDeviceToHost));
  // CHECK_CUDA(cudaMemcpy(h_dst_u, d_dst_u, dst_u_size, cudaMemcpyDeviceToHost));
  // CHECK_CUDA(cudaMemcpy(h_dst_v, d_dst_v, dst_v_size, cudaMemcpyDeviceToHost));
  std::ofstream output_file(output_image_path, std::ios::binary);
  if (!output_file.is_open()) {
    std::cerr << "Failed to open output file: " << output_image_path << std::endl;
    return;
  }
  output_file.write(reinterpret_cast<char*>(h_dst.data()), h_dst.size());

  // 释放device端内存
  CHECK_CUDA(cudaFree(d_src_y));
  CHECK_CUDA(cudaFree(d_src_u));
  CHECK_CUDA(cudaFree(d_src_v));
  CHECK_CUDA(cudaFree(d_dst_y));
  CHECK_CUDA(cudaFree(d_dst_u));
  CHECK_CUDA(cudaFree(d_dst_v));

}

void npp_convert_example() {
  // 生成一个简单的RGB图像
  const int width = 4;
  const int height = 4;
  const int channels = 3;
  const int src_step = width * channels * sizeof(Npp8u); // 8-bit unsigned每像素3字节
  const int dst_step = width * channels * sizeof(Npp32f); // 32-bit float每像素12字节
  std::vector<Npp8u> h_src = {
    11, 22, 33,   255, 0, 0,   0, 255, 0,   0, 0, 255,
    255, 255, 0,   255, 0, 255,   0, 255, 255,   255, 255, 255,
    128, 128, 128,   64, 64, 64,   192, 192, 192,   32, 32, 32,
    16, 16, 16,   240, 240, 240,   8, 8, 8,   248, 248, 248
  };
  std::vector<Npp32f> h_dst(width * height * channels, 0.0f);

  // 分配device端内存并将数据从host端拷贝到device端
  Npp8u* d_src = nullptr;
  Npp32f* d_dst = nullptr;
  CHECK_CUDA(cudaMalloc(&d_src, h_src.size() * sizeof(Npp8u)));
  CHECK_CUDA(cudaMalloc(&d_dst, h_dst.size() * sizeof(Npp32f)));
  CHECK_CUDA(cudaMemcpy(d_src, h_src.data(), h_src.size() * sizeof(Npp8u), cudaMemcpyHostToDevice));
  NppiSize roi_size = { width, height };

  /**
   * @brief 将图像的像素数据类型做转换，例如从8-bit unsigned转换为32-bit float
   * @param pSrc 源图像数据指针
   * @param nSrcStep 源图像每行的字节数(stride)
   * @param pDst 目标图像数据指针
   * @param nDstStep 目标图像每行的字节数(stride)
   * @param oSizeROI roi区域的尺寸
   */
  CHECK_NPP(nppiConvert_8u32f_C3R(d_src, src_step, d_dst, dst_step, roi_size));

  // 将数据从device端拷贝回host端并打印部分结果
  CHECK_CUDA(cudaMemcpy(h_dst.data(), d_dst, h_dst.size() * sizeof(Npp32f), cudaMemcpyDeviceToHost));
  std::cout << "After nppiConvert, pixel (0,0): R G B = "
            << h_dst[0] << " " << h_dst[1] << " " << h_dst[2] << std::endl;

  // 释放device端内存
  CHECK_CUDA(cudaFree(d_src));
  CHECK_CUDA(cudaFree(d_dst));

}

void npp_mulc_example() {
  // 生成一个简单的RGB图像
  const int width = 4;
  const int height = 4;
  const int channels = 3;
  const int src_step = width * channels * sizeof(Npp32f); // 32-bit float每像素12字节
  std::vector<Npp32f> h_src = {
    1.0f, 2.0f, 3.0f,
    4.0f, 5.0f, 6.0f,
    7.0f, 8.0f, 9.0f,
    10.0f, 11.0f, 12.0f,
    13.0f, 14.0f, 15.0f,
    16.0f, 17.0f, 18.0f,
    19.0f, 20.0f, 21.0f,
    22.0f, 23.0f, 24.0f
  };
  std::vector<Npp32f> h_dst(width * height * channels, 0.0f);

  // 分配device端内存并将数据从host端拷贝到device端
  Npp32f* d_src = nullptr;
  Npp32f* d_dst = nullptr;
  CHECK_CUDA(cudaMalloc(&d_src, h_src.size() * sizeof(Npp32f)));
  CHECK_CUDA(cudaMalloc(&d_dst, h_dst.size() * sizeof(Npp32f)));
  CHECK_CUDA(cudaMemcpy(d_src, h_src.data(), h_src.size() * sizeof(Npp32f), cudaMemcpyHostToDevice));
  NppiSize roi_size = { width, height };
  Npp32f alpha = 2.0f;
  std::vector<Npp32f> h_constants = { alpha * 2.0f, alpha, alpha * 0.5f }; // 每个通道的常数因子

  /**
   * @brief 对图像的每个像素值乘以一个常数因子，例如对图像进行亮度调整
   * @param pSrc 源图像数据指针
   * @param nSrcStep 源图像每行的字节数(stride)
   * @param aConstants 固定大小常数因子数组指针，长度等于图像的通道数
   * @param pDst 目标图像数据指针
   * @param nDstStep 目标图像每行的字节数(stride)
   * @param oSizeROI roi区域的尺寸
   *
   * @note MulC其实是Mul跟Constant的缩写
   */
   CHECK_NPP(nppiMulC_32f_C3R(d_src, src_step, h_constants.data(), d_dst, src_step, roi_size));

  // 将数据从device端拷贝回host端并打印部分结果
  CHECK_CUDA(cudaMemcpy(h_dst.data(), d_dst, h_dst.size() * sizeof(Npp32f), cudaMemcpyDeviceToHost));
  std::cout << "After nppiMulC, pixel (0,0): R G B = "
            << h_dst[0] << " " << h_dst[1] << " " << h_dst[2] << std::endl;
  
  // 释放device端内存
  CHECK_CUDA(cudaFree(d_src));
  CHECK_CUDA(cudaFree(d_dst));
}

int main() {
  std::cout << "=============== NPP Sample Example ===============" << std::endl;
  npp_sample_example();
  std::cout << "=============== NPP Resize Example ===============" << std::endl;
  npp_resize_example();
  std::cout << "=============== NPP Crop Example ===============" << std::endl;
  npp_crop_example();
  std::cout << "=============== NPP Convert Example ===============" << std::endl;
  npp_convert_example();
  std::cout << "=============== NPP MulC Example ===============" << std::endl;
  npp_mulc_example();

  return 0;
}
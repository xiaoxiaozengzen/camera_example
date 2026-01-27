#include <iostream>
#include <vector>
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

int main() {
  cudaStream_t stream = nppGetStream();
  if(stream) {
    std::cout << "Current NPP CUDA Stream: " << stream << std::endl;
  } else {
    std::cout << "Current NPP CUDA Stream: Default Stream (0)" << std::endl;
  }

  NppStreamContext nppStreamCtx;
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

  unsigned char *d_src=nullptr, *d_dst=nullptr;
  CHECK_CUDA(cudaMalloc(&d_src, h_src.size()));
  CHECK_CUDA(cudaMalloc(&d_dst, h_dst.size()));
  CHECK_CUDA(cudaMemcpy(d_src, h_src.data(), h_src.size(), cudaMemcpyHostToDevice));

  NppiSize roi = { w, h };

  /**
   * @brief 给一个8U3C的RGB图像，源图按照ROI区域和给定源图步长，拷贝到目标图像的对应区域，目标图按照给定的目标图步长存储
   * @param pSrc 源图像数据指针
   * @param nSrcStep 源图像每行的字节数（stride）
   * @param pDst 目标图像数据指针
   * @param nDstStep 目标图像每行的字节数（stride）
   * @param oSizeROI 图像的宽度
   */
  CHECK_NPP(nppiCopy_8u_C3R(d_src, srcStep, d_dst, dstStep, roi));

  // 将数据从device拷贝回host并打印部分结果
  CHECK_CUDA(cudaMemcpy(h_dst.data(), d_dst, h_dst.size(), cudaMemcpyDeviceToHost));
  std::cout << "After nppiCopy, pixel (0,1): R G B = "
            << int(h_dst[3]) << " " << int(h_dst[4]) << " " << int(h_dst[5]) << std::endl;

  NppiSize mask = {3, 3};
  NppiPoint anchor = {1, 1}; // center of the kernel
  
  /**
   * @brief 对图像执行盒式均值滤波，用指定大小的矩形核对每个像素做局部平均，达到模糊/降噪效果
   * @param pSrc 源图像数据指针
   * @param nSrcStep 源图像每行的字节数（stride）
   * @param pDst 目标图像数据指针
   * @param nDstStep 目标图像每行的字节数（stride）
   * @param oSizeROI 图像的宽度区域
   * @param oMaskSize 矩形核的大小
   * @param oAnchor 矩形核的锚点
   *
   * @note 矩形核的数值均为1/(mask.width * mask.height)
   */
  CHECK_NPP(nppiFilterBox_8u_C3R(d_src, srcStep, d_dst, dstStep, roi, mask, anchor));

  CHECK_CUDA(cudaMemcpy(h_dst.data(), d_dst, h_dst.size(), cudaMemcpyDeviceToHost));
  std::cout << "After box filter, pixel (0,0): R G B = "
            << int(h_dst[0]) << " " << int(h_dst[1]) << " " << int(h_dst[2]) << std::endl;

  cudaFree(d_src);
  cudaFree(d_dst);
  return 0;
}
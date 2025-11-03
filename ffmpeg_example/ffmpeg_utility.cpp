#include <iostream>
#include <string>
#include <fstream>
#include <algorithm>
#include <cerrno>
#include <iomanip>

extern "C" {
#include "libavformat/avformat.h"
#include "libavcodec/avcodec.h"
#include "libavutil/pixfmt.h"
#include "libavutil/avutil.h"
#include "libavutil/pixdesc.h"
#include "libavutil/opt.h"
}

void codec_name() {
    std::string h265_codec_name;
    h265_codec_name = avcodec_get_name(AVCodecID::AV_CODEC_ID_HEVC);
    std::cout << "H.265 codec name: " << h265_codec_name << std::endl;

    std::string h264_codec_name;
    h264_codec_name = avcodec_get_name(AVCodecID::AV_CODEC_ID_H264);
    std::cout << "H.264 codec name: " << h264_codec_name << std::endl;

    AVCodec *codec = avcodec_find_decoder_by_name(h265_codec_name.c_str());
    if(codec) {
        std::cout << "Found decoder for H.265: " << codec->name << std::endl;
    } else {
        std::cout << "Decoder for H.265 not found!" << std::endl;
    }
}

/**
 * AV_HWDEVICE_TYPE_NONE  不适用硬件加速，采用软解码(CPU解码)
 * AV_HWDEVICE_TYPE_VDPAU  Video Decode and Presentation API for Unix，主要用于NVIDIA GPU的视频解码加速，主要在Linux系统上使用
 *                         改API比较老旧，现在较少使用
 * AV_HWDEVICE_TYPE_CUDA  NVIDIA的通用计算架构CUDA，需要平台支持CUDA，主要用于NVIDIA GPU的视频解码加速
 * AV_HWDEVICE_TYPE_VAAPI  Video Acceleration API，主要用于Intel集成显卡的视频解码加速，在Linux系统上较为常见
 * AV_HWDEVICE_TYPE_DXVA2  DirectX Video Acceleration 2，主要在Windows系统上使用，利用GPU进行视频解码加速
 * AV_HWDEVICE_TYPE_QSV   Intel Quick Sync Video，主要用于Intel处理器的快速视频编码和解码
 * AV_HWDEVICE_TYPE_VIDEOTOOLBOX  主要在macOS和iOS上使用，利用硬件加速视频编码和解码
 * AV_HWDEVICE_TYPE_D3D11VA  Direct3D 11 Video Acceleration，主要在Windows系统上使用，利用Direct3D 11进行视频解码加速
 * AV_HWDEVICE_TYPE_DRM  Direct Rendering Manager，主要在Linux系统上使用，用于硬件加速的图形显示
 * AV_HWDEVICE_TYPE_OPENCL  Open Computing Language，可用于通用的并行计算，也可在某些情况下用于视频处理加速
 * AV_HWDEVICE_TYPE_MEDIACODEC  Android平台的硬件加速视频编解码器
 * AV_HWDEVICE_TYPE_VULKAN  Vulkan API，可用于高性能图形和计算应用程序，也可用于视频处理加速
 */
void hw_device() {
    std::string vdpua_str = av_hwdevice_get_type_name(AVHWDeviceType::AV_HWDEVICE_TYPE_VDPAU);
    std::cout << "AV_HWDEVICE_TYPE_VDPAU: " << vdpua_str << std::endl;

    std::string cuda_str = av_hwdevice_get_type_name(AVHWDeviceType::AV_HWDEVICE_TYPE_CUDA);
    std::cout << "AV_HWDEVICE_TYPE_CUDA: " << cuda_str << std::endl;

    AVHWDeviceType device_type = av_hwdevice_find_type_by_name("cuda");
    if(device_type != AV_HWDEVICE_TYPE_NONE) {
        std::cout << "Found device type for 'cuda': " << av_hwdevice_get_type_name(device_type) << std::endl;
    } else {
        std::cout << "Device type for 'cuda' not found!" << std::endl;
    }

    device_type = av_hwdevice_find_type_by_name("videotoolbox");
    if(device_type != AV_HWDEVICE_TYPE_NONE) {
        std::cout << "Found device type for 'videotoolbox': " << av_hwdevice_get_type_name(device_type) << std::endl;
    } else {
        std::cout << "Device type for 'videotoolbox' not found!" << std::endl;
    }
    device_type = av_hwdevice_find_type_by_name("VAPPI");
    if(device_type != AV_HWDEVICE_TYPE_NONE) {
        std::cout << "Found device type for 'VAPPI': " << av_hwdevice_get_type_name(device_type) << std::endl;
    } else {
        std::cout << "Device type for 'VAPPI' not found!" << std::endl;
    }

    AVCodecContext *codec_ctx = avcodec_alloc_context3(nullptr);
    if(!codec_ctx) {
        std::cout << "Could not allocate codec context!" << std::endl;
        return;
    }

    AVCodec *codec = avcodec_find_decoder(AVCodecID::AV_CODEC_ID_H265);
    if(codec) {
        std::cout << "Found decoder for H.265: " << codec->name << std::endl;
    } else {
        std::cout << "Decoder for H.265 not found!" << std::endl;
        return;
    }

    int ret = avcodec_open2(codec_ctx, codec, nullptr);
    if(ret < 0) {
        char errbuf[AV_ERROR_MAX_STRING_SIZE] = {0};
        av_strerror(ret, errbuf, sizeof(errbuf));
        std::cout << "Could not open codec, because: "
                  << ret << " -> " << errbuf << std::endl;
        avcodec_free_context(&codec_ctx);
        return;
    }

    for(int i = 0;; i++) {
        /**
        * struct AVCodecHWConfig {
        *  enum AVPixelFormat pix_fmt;  // 像素格式
        *  int methods;  // 支持的硬件加速方法的位掩码，一系列的AV_CODEC_HW_CONFIG_METHOD_*常量的组合
        *  enum AVHWDeviceType device_type;  // 硬件设备类型，
        *                                    必须设置AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX跟AV_CODEC_HW_CONFIG_METHOD_HW_FRAMES_CTX，改参数才会被使用
        * }
        */
        const AVCodecHWConfig *config = avcodec_get_hw_config(codec, i);
        if(config) {
            std::cout << "Found hardware config for H.265: " << av_hwdevice_get_type_name(config->device_type) << std::endl;
            std::cout << "  Pixel Format: " << av_get_pix_fmt_name(config->pix_fmt) << std::endl;
            std::cout << "  Methods: ";
            if(config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX) {
                std::cout << "HW_DEVICE_CTX ";
            }
            if(config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_FRAMES_CTX) {
                std::cout << "HW_FRAMES_CTX ";
            }
            std::cout << std::endl;
        } else {
            break;
        }
    }

    /**
     * struct AVBufferRef {
     *  AVBuffer* buffer;  // 指向实际缓冲区的指针
     *  uint8_t* data;  // 指向缓冲区数据的指针
     *  int size;  // data指针所指向的数据大小
     * }
     *
     * struct AVBuffer {
     *  uint8_t* data;  // 指向缓冲区数据的指针
     *  int size;  // data指针所指向的数据大小
     *  atomic_uint refcount;  // 引用计数，用于管理缓冲区的生命周期
     *  ... // 其他成员省略
     * }
     */
    AVBufferRef *hw_device_ctx = nullptr;

    ret = av_hwdevice_ctx_create(&hw_device_ctx,
                                   AVHWDeviceType::AV_HWDEVICE_TYPE_VAAPI,
                                   nullptr,
                                   nullptr,
                                   0);
    if(ret < 0) {
        char errbuf[AV_ERROR_MAX_STRING_SIZE] = {0};
            av_strerror(ret, errbuf, sizeof(errbuf));
            std::cout << "Could not create HW device context for VAAPI, because: "
                    << ret << " -> " << errbuf << std::endl;
    } else {
        std::cout << "Created HW device context for VAAPI." << std::endl;
        std::cout << "HW device context buffer: " << hw_device_ctx->buffer << std::endl;
        std::cout << "HW device context data: " << reinterpret_cast<void*>(hw_device_ctx->data) << std::endl;
        std::cout << "HW device context size: " << hw_device_ctx->size << std::endl;
    }


    if(hw_device_ctx) {
        codec_ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);
        if(!codec_ctx->hw_device_ctx) {
            std::cout << "Could not set HW device context to codec context!" << std::endl;
        } else {
            std::cout << "Set HW device context to codec context." << std::endl;
        }
    }

    /**
     * @brief 设置选项值
     * @note 可以通过ffmpeg -h encoder=libx264命令查看libx264编码器支持的私有选项
     */
    ret = av_opt_set(codec_ctx->priv_data, "tune", "zerolatency", 0);
    if(ret < 0) {
        char errbuf[AV_ERROR_MAX_STRING_SIZE] = {0};
        av_strerror(ret, errbuf, sizeof(errbuf));
        std::cout << "Could not set tune option, because: "
                  << ret << " -> " << errbuf << std::endl;
    } else {
        std::cout << "Set tune option to zerolatency." << std::endl;
    }

    av_buffer_unref(&hw_device_ctx);
    avcodec_close(codec_ctx);
    avcodec_free_context(&codec_ctx);
}

int main() {
    std::cout << " ============== Codec Name ==============" << std::endl;
    codec_name();
    std::cout << " ============== HW Device Type ==============" << std::endl;
    hw_device();
    return 0;
}
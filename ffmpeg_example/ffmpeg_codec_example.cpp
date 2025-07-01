#include <iostream>
#include <string>
#include <fstream>
#include <algorithm>

extern "C" {
#include "libavformat/avformat.h"
#include "libavcodec/avcodec.h"
#include "libavutil/pixfmt.h"
#include "libavutil/avutil.h"
#include "libavutil/pixdesc.h"
}

#include <opencv2/opencv.hpp>

void yuv2jpg(std::string yuv_file);

/**
 * @brief This function is used to open a video file and print its metadata.
 * 
 * 解码流程：
 *   1.使用demuxer打开视频文件，获取AVFormatContext。
 *     - avformat_open_input() 打开输入流并读取其头部信息。
 *     - avformat_find_stream_info() 查找流信息。
 *   2.codec选择解码器
 *     - avcodec_find_decoder() 查找解码器。
 *     - avcodec_alloc_context3() 分配解码器上下文。
 *     - avcodec_parameters_to_context() 将流的编解码参数复制到解码器上下文。
 *     - avcodec_open2() 打开解码器。
 *   3.读取数据包
 *     - av_read_frame() 从输入流中读取数据包。
 *   4.解码数据包
 *     - avcodec_send_packet() 将数据包发送到解码器。
 *     - avcodec_receive_frame() 从解码器接收解码后的帧。
 *   5.处理解码后的帧
 *     - 可以对解码后的帧进行处理，如显示、保存等。
 */
int open_fun() {
    std::string mp4_file = "/mnt/workspace/cgz_workspace/Exercise/camera_example/input/video.mp4";

    /**
     * @brief 封装格式上下文。可以理解为 MP4 文件的容器格式。
     */
    AVFormatContext *format_context = nullptr;
    format_context = avformat_alloc_context();
    if(!format_context) {
        std::cerr << "Could not allocate format context, because: " << AVERROR(ENOMEM) << std::endl;
        return -1;
    }

    /**
     * int avformat_open_input(AVFormatContext **ps, const char *url, const AVInputFormat *fmt, AVDictionary **options);
     * 
     * @brief 打开一个输入流并读取其头部信息。codec并未被打开。注意：一定要调用avformat_close_input().
     * @param ps 指向 AVFormatContext 的指针的地址。这个指针将被填充为打开的格式上下文。
     * @param url 输入流的 URL 或文件名。
     * @param fmt 指定输入格式，强制使用特定的输入格式。如果为 NULL，FFmpeg 将自动检测格式。
     * @param options 指向 AVDictionary 的指针，用于设置打开输入流时的选项。
     * @return 成功是返回 0；失败时间：释放掉ps，并设置其指向null，并返回一个负数AVERROR
     */
    int ret = avformat_open_input(&format_context, mp4_file.c_str(), nullptr, nullptr);
    if(ret < 0) {
        std::cerr << "Could not open input file '" << mp4_file << std::endl;
        avformat_free_context(format_context);
        return -1;
    }

    std::string url(format_context->url);
    std::int32_t start_time = format_context->start_time;
    std::int64_t duration = format_context->duration;
    std::uint32_t nb_streams = format_context->nb_streams;
    std::uint64_t bit_rate = format_context->bit_rate;
    std::string iformat_name(format_context->iformat->name);
    std::string iformat_long_name(format_context->iformat->long_name);

    std::cout << "AVFormatContex url: " << url << std::endl;
    std::cout << "AVFormatContex AV_NOPTS_VALUE: " << AV_NOPTS_VALUE << std::endl;  // 表示无法正确获取数据
    std::cout << "AVFormatContex start_time: " << start_time << std::endl;
    std::cout << "AVFormatContex AV_TIME_BASE: " << AV_TIME_BASE << std::endl;

    /**
     * duration: 视频的持续时间，单位为 AV_TIME_BASE（通常是微秒）。
     * FFmpeg 从 mp4 里面的 duration读取的
     * MP4 标准定义 mvhd 里面的 duration 以最长的流为准，这个 duration 可能是音频流的时长，也可能是视频流的时长。
     */
    std::cout << "AVFormatContex duration: " << duration / double(AV_TIME_BASE) << std::endl;
    std::cout << "AVFormatContex nb_streams: " << nb_streams << std::endl;
    for(std::size_t i = 0; i < nb_streams; ++i) {
        /**
         * @brief AVStream 是 FFmpeg 中的一个结构体，表示多媒体流（如音频或视频流）。
         */
        AVStream *stream = format_context->streams[i];
        if(stream) {
            std::cout << "Stream " << i 
                      << ": index: " << stream->index
                      << ", id: " << stream->id
                      << ", start_time: " << stream->start_time
                      << ", duration: " << stream->duration
                      << ", nb_frames: " << stream->nb_frames
                      << ", disposition: " << stream->disposition
                      << ", pts_wrap_bits: " << stream->pts_wrap_bits
                      << ": codec_type: " << av_get_media_type_string(stream->codecpar->codec_type)
                      << ", codec_id: " << avcodec_get_name(stream->codecpar->codec_id) 
                      << std::endl;
        }
    }
    std::cout << "AVFormatContex bit_rate: " << bit_rate / 1000 << " kbps" << std::endl;
    std::cout << "AVFormatContex iformat_name: " << iformat_name << std::endl;
    std::cout << "AVFormatContex iformat_long_name: " << iformat_long_name << std::endl;


    /**
     * @brief AVCodecContext 是 FFmpeg 中的一个结构体，表示编解码器上下文。
     */
    AVCodecContext *codec_context = nullptr;
    codec_context = avcodec_alloc_context3(nullptr);
    if(!codec_context) {
        std::cerr << "Could not allocate AVCodecContext, because: " << AVERROR(ENOMEM) << std::endl;
        avformat_close_input(&format_context);
        avformat_free_context(format_context);
        return -1;
    }

    /**
     * @brief AVCodecParameters 是 FFmpeg 中的一个结构体，表示编解码器参数。
     * 
     * codecpar 包含了编解码器的各种参数，如编码格式、分辨率、比特率等。
     */
    AVCodecParameters *codecpar = format_context->streams[0]->codecpar;
    std::cout << "AVCodecParameters codec_type: " << av_get_media_type_string(codecpar->codec_type) << std::endl;
    std::cout << "AVCodecParameters odec_id: " << avcodec_get_name(codecpar->codec_id) << std::endl;
    std::cout << "AVCodecParameters codec_tag: " << codecpar->codec_tag << std::endl;
    std::cout << "AVCodecParameters format: " << codecpar->format << std::endl;
    std::cout << "AVCodecParameters bit_rate: " << codecpar->bit_rate / 1000 << " kbps" << std::endl;
    std::cout << "AVCodecParameters profile: " << codecpar->profile << std::endl;
    std::cout << "AVCodecParameters level: " << codecpar->level << std::endl;
    std::cout << "AVCodecParameters width: " << codecpar->width << std::endl;
    std::cout << "AVCodecParameters height: " << codecpar->height << std::endl;
    ret = avcodec_parameters_to_context(codec_context, codecpar);
    if(ret < 0) {
        std::cerr << "Could not copy codec parameters to context, because: " << AVERROR(ret) << std::endl;
        avcodec_free_context(&codec_context);
        avformat_close_input(&format_context);
        avformat_free_context(format_context);
        return -1;
    }

    /**
     * @brief AVCodec 是 FFmpeg 中的一个结构体，表示编解码器。
     */
    AVCodec *codec = avcodec_find_decoder(codec_context->codec_id);
    if(!codec) {
        std::cerr << "Could not find codec for codec_id: " << avcodec_get_name(codec_context->codec_id) << std::endl;
        avcodec_free_context(&codec_context);
        avformat_close_input(&format_context);
        avformat_free_context(format_context);
        return -1;
    }
    std::cout << "AVCodec name: " << codec->name << std::endl;
    std::cout << "AVCodec long name: " << codec->long_name << std::endl;

    /**
     * int avcodec_open2(AVCodecContext *avctx, const AVCodec *codec, AVDictionary **options);
     * 
     * @brief 初始化AVCodecContext来使用给定的Codec。使用该函数之前，先调用avcode_alloc_context3()分配AVCodecContext。
     * @param avctx 指向 AVCodecContext 的指针。
     * @param codec 指向 AVCodec 的指针，表示要使用的编解码器。
     * @param options 指向 AVDictionary 的指针，用于设置编解器选项。
     * @return 成功时返回 0，失败时返回一个负数错误代码。
     */
    ret = avcodec_open2(codec_context, codec, nullptr);
    if(ret < 0) {
        std::cerr << "Could not open codec, because: " << AVERROR(ret) << std::endl;
        avcodec_free_context(&codec_context);
        avformat_close_input(&format_context);
        avformat_free_context(format_context);
        return -1;
    }


    /**
     * @brief AVPacket 是 FFmpeg 中的一个结构体，表示多媒体数据包。
     * 
     * AVPacket 保存编码后的压缩数据，通常是音频或视频帧。
     */
    AVPacket* packet = nullptr;
    packet = av_packet_alloc();
    if(!packet) {
        std::cerr << "Could not allocate AVPacket, because: " << AVERROR(ENOMEM) << std::endl;
        avcodec_free_context(&codec_context);
        avformat_close_input(&format_context);
        avformat_free_context(format_context);
        return -1;
    }

    /**
     * @brief AVFrame 是 FFmpeg 中的一个结构体，表示解码后的帧数据。
     */
    AVFrame *frame = av_frame_alloc();
    if(!frame) {
        std::cerr << "Could not allocate AVFrame, because: " << AVERROR(ENOMEM) << std::endl;
        av_packet_free(&packet);
        avcodec_free_context(&codec_context);
        avformat_close_input(&format_context);
        avformat_free_context(format_context);
        return -1;
    }

    int packet_count = 0;
    int frame_count = 0;
    while(true) {
        /**
         * @brief av_read_frame() 函数用于从输入流中读取一个数据包。
         * 
         * @return 成功时返回 0，失败时返回一个负数错误代码。
         */
        ret = av_read_frame(format_context, packet);
        if(ret < 0) {
            std::cerr << "Could not read frame from input file, because: " << AVERROR(ret) << std::endl;
            break;
        }
        if(packet->stream_index == 1) {
            continue;  // 跳过音频流，只处理视频流
        }
        if(packet_count == 1) {
            std::cout << "------------------Packet count: " << packet_count << std::endl;
            std::cout << "Packet pts: " << packet->pts << std::endl;
            std::cout << "Packet dts: " << packet->dts << std::endl;
            std::cout << "Packet size: " << packet->size << std::endl;
            std::cout << "Packet stream index: " << packet->stream_index << std::endl;
            std::cout << "Packet flags: " << packet->flags << std::endl;
            std::cout << "Packet duration: " << packet->duration << std::endl;
            std::cout << "Packet pos: " << packet->pos << std::endl;
            std::cout << "Packet side_data_elems: " << packet->side_data_elems << std::endl;
        }

        /**
         * int avcodec_send_packet(AVCodecContext *avctx, const AVPacket *avpkt);
         * 
         * @brief 将 AVPacket 发送到解码器进行解码。
         * @param avctx 指向 AVCodecContext 的指针，表示解码器上下文。
         * @param avpkt 指向 AVPacket 的指针，表示要解码的数据包。
         * @return 成功时返回 0，失败时返回一个负数错误代码。
         * 
         * @note avctx必须经过avcodec_open2()函数初始化。
         */
        ret = avcodec_send_packet(codec_context, packet);
        if(ret < 0) {
            std::cerr << "Could not send packet to decoder, because: " << AVERROR(ret) << std::endl;
            break;
        }

        // 释放 AVPacket
        av_packet_unref(packet);
        packet_count++;

        while(true) {
            frame_count++;

            /**
             * int avcodec_receive_frame(AVCodecContext *avctx, AVFrame *frame);
             * 
             * @brief 返回解码后的 AVFrame。
             * @param avctx 指向 AVCodecContext 的指针，表示解码器上下文。
             * @param frame 指向 AVFrame 的指针，表示解码后的帧数据。
             * @return 成功时返回 0，失败时返回一个负数错误代码。
             * 
             * @note 函数会自动调用 av_frame_unref(frame) 来释放之前的帧数据。
             * 
             */
            ret = avcodec_receive_frame(codec_context, frame);
            if(ret < 0) {
                if(ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                    // EAGAIN 表示没有更多的帧可供解码，EOF 表示到达了文件末尾
                    break;
                } else {
                    std::cerr << "Could not receive frame from decoder, because: " << AVERROR(ret) << std::endl;
                    break;
                }
            }
            if(frame_count == 1) {
                /**
                 * @brief 打印解码后的帧信息
                 * 
                 * @note data指向解码后picture的planes数据。
                 * @note linesize表示每个平面（plane）的行大小，单位是字节。
                 */
                std::cout << "Frame data: " << reinterpret_cast<void*>(frame->data[0]) << std::endl;
                std::cout << "Frame linesize[0]: " << frame->linesize[0] << std::endl;
                std::cout << "Frame linesize[1]: " << frame->linesize[1] << std::endl;
                std::cout << "Frame linesize[2]: " << frame->linesize[2] << std::endl;
                std::cout << "Frame linesize[3]: " << frame->linesize[3] << std::endl;
                std::cout << "Frame width: " << frame->width << std::endl;
                std::cout << "Frame height: " << frame->height << std::endl;
                std::cout << "Frame format: " << frame->format << std::endl;
                std::cout << "Frame picture_type: " << av_get_picture_type_char(frame->pict_type) << std::endl;
                std::cout << "Frame pts: " << frame->pts << std::endl;
                std::cout << "Frame pkt_dts: " << frame->pkt_dts << std::endl;
                std::cout << "Frame quality: " << frame->quality << std::endl;
                std::cout << "Frame repeat_pict: " << frame->repeat_pict << std::endl;
                std::cout << "Frame color_range: " << frame->color_range << std::endl;

                /**
                 * format == 0，这是 YUV420P 格式
                 * @brief 其值跟枚举 AVPixelFormat 对应。
                 */
                AVPixelFormat pix_fmt = (AVPixelFormat)frame->format;
                const AVPixFmtDescriptor *pix_fmt_desc = av_pix_fmt_desc_get(pix_fmt);
                std::cout << "AVPixFmtDescriptor name: " << pix_fmt_desc->name << std::endl;
                std::cout << "AVPixFmtDescriptor nb_components: " << (uint16_t)pix_fmt_desc->nb_components << std::endl;
                std::cout << "AVPixFmtDescriptor log2_chroma_w: " << (uint16_t)pix_fmt_desc->log2_chroma_w << std::endl;
                std::cout << "AVPixFmtDescriptor log2_chroma_h: " << (uint16_t)pix_fmt_desc->log2_chroma_h << std::endl;
                std::cout << "AVPixFmtDescriptor flags: " << pix_fmt_desc->flags << std::endl;
                if(pix_fmt_desc->flags & AV_PIX_FMT_FLAG_ALPHA) {
                    std::cout << "AVPixFmtDescriptor has alpha channel." << std::endl;
                } else {
                    std::cout << "AVPixFmtDescriptor does not have alpha channel." << std::endl;
                }
                if(pix_fmt_desc->flags & AV_PIX_FMT_FLAG_RGB) {
                    std::cout << "AVPixFmtDescriptor is RGB format." << std::endl;
                } else {
                    std::cout << "AVPixFmtDescriptor is not RGB format." << std::endl;
                }
                /**
                 * @brief AV_PIX_FMT_FLAG_BITSTREAM 表示该像素格式是一个比特流格式，即连续的
                 */
                if(pix_fmt_desc->flags & AV_PIX_FMT_FLAG_BITSTREAM) {
                    std::cout << "AVPixFmtDescriptor is a bitstream format." << std::endl;
                } else {
                    std::cout << "AVPixFmtDescriptor is not a bitstream format." << std::endl;
                }
                /**
                 * @brief AV_PIX_FMT_FLAG_PLANAR 表示该像素格式是一个平面格式，即每个颜色通道的数据存储在单独的平面中。
                 * 
                 * @note yuv420p中的p就是表示planner格式。不带p的表示packed格式。
                 */
                if(pix_fmt_desc->flags & AV_PIX_FMT_FLAG_PLANAR) {
                    std::cout << "AVPixFmtDescriptor is a planar format." << std::endl;
                } else {
                    std::cout << "AVPixFmtDescriptor is not a planar format." << std::endl;
                }

                /**
                 * @brief AVComponentDescriptor 是 FFmpeg 中的一个结构体，表示像素如何被打包的
                 * 
                 * 1.如果format还有1或者2个component，那么luma是0
                 * 2.如果format有3或者4个component
                 *   - 如果RGB被设置，那么0是R，1是G，2是B
                 *   - 如果没有RGB被设置，那么0是Y，1是U，2是V
                 * 
                 * @note plane 表示该组件所在的平面，可以理解成行
                 * @note component可以理解为分量，比如 YUV 的 Y、U、V 分量。
                 */
                AVComponentDescriptor comp = pix_fmt_desc->comp[0];
                std::cout << "AVComponentDescriptor plane: " << (uint16_t)comp.plane << std::endl;
                std::cout << "AVComponentDescriptor step: " << comp.step << std::endl;
                std::cout << "AVComponentDescriptor depth: " << (uint16_t)comp.depth << std::endl;
                std::cout << "AVComponentDescriptor offset: " << (uint16_t)comp.offset << std::endl;
                std::cout << "AVComponentDescriptor shift: " << (uint16_t)comp.shift << std::endl;
                AVComponentDescriptor comp_u = pix_fmt_desc->comp[1];
                std::cout << "AVComponentDescriptor plane_u: " << (uint16_t)comp_u.plane << std::endl;
                std::cout << "AVComponentDescriptor step_u: " << comp_u.step << std::endl;
                std::cout << "AVComponentDescriptor depth_u: " << (uint16_t)comp_u.depth << std::endl;
                std::cout << "AVComponentDescriptor offset_u: " << (uint16_t)comp_u.offset << std::endl;
                std::cout << "AVComponentDescriptor shift_u: " << (uint16_t)comp_u.shift << std::endl;

                /**
                 * @brief 生成一个yuv的图像
                 */
                std::string output_path = "/mnt/workspace/cgz_workspace/Exercise/camera_example/output";
                std::string yuv_image_name = "/frame_" + std::to_string(frame_count) + ".yuv";
                std::string yuv_image_path = output_path + yuv_image_name;
                std::ofstream yuv_file(yuv_image_path, std::ios::binary);
                if(!yuv_file) {
                    std::cerr << "Could not open output file '" << yuv_image_path << "' for writing." << std::endl;
                } else {
                    yuv_file.write(reinterpret_cast<const char*>(frame->data[0]), frame->width * frame->height);
                    yuv_file.write(reinterpret_cast<const char*>(frame->data[1]), frame->width / 2 * frame->height / 2);
                    yuv_file.write(reinterpret_cast<const char*>(frame->data[2]), frame->width / 2 * frame->height / 2);   
                }
                yuv_file.close();
                yuv2jpg(yuv_image_path);
            }
        }
        
    }

    std::cout << "Total packets read: " << packet_count << std::endl;
    std::cout << "Total frames decoded: " << frame_count << std::endl;

    // 关闭 AVPacket
    av_packet_free(&packet);
    // 释放 AVFrame
    av_frame_free(&frame);
    // 关闭解码器
    avcodec_free_context(&codec_context);
    // 关闭输入文件
    avformat_close_input(&format_context);
    // 释放格式上下文
    avformat_free_context(format_context);

    return 0;
}

/**
 * @brief This function is to use demuxer
 * @note demuxer跟codec的关系：
 *        * demuxer 解析多媒体文件格式，并把文件分成不同的流（如音频流和视频流）。
 *        * codec 负责编码和解码具体流中的数据，只处理流中的packet
 */
int demuxer_example() {
    std::string mp4_file = "/mnt/workspace/cgz_workspace/Exercise/camera_example/input/video.mp4";

    AVFormatContext *format_context = nullptr;
    format_context = avformat_alloc_context();
    if(!format_context) {
        std::cerr << "Could not allocate format context, because: " << AVERROR(ENOMEM) << std::endl;
        return -1;
    }

    /**
     * @brief AVDictionary 是 FFmpeg 中的一个结构体，用于存储键值对形式的选项。
     * struct AVDictionary {
     *   AVDictionaryEntry *elems; // 键值对数组
     *   int count;                // 键值对数量
     * }
     * 
     * struct AVDictionaryEntry {
     *   char *key;                // 键
     *   char *value;              // 值
     * }
     */
    AVDictionary *options = nullptr;
    av_dict_set(&options, "probesize", "32", AV_DICT_MATCH_CASE);
    av_dict_set(&options, "timeout", "10", AV_DICT_MATCH_CASE);

    AVDictionaryEntry *entry = nullptr;
    entry = av_dict_get(options, "", nullptr, AV_DICT_IGNORE_SUFFIX);
    if(entry) {
        std::cout << "Dictionary entry 1: key = " << entry->key << ", value = " << entry->value << std::endl;
        std::cout << "Dictionary entry 2: key = " << (entry+1)->key << ", value = " << (entry+1)->value << std::endl;
    } else {
        std::cout << "No dictionary entry found." << std::endl;
    }

    int ret = avformat_open_input(&format_context, mp4_file.c_str(), nullptr, &options);
    if(ret < 0) {
        std::cerr << "Could not open input file '" << mp4_file << "' because: " << AVERROR(ENOMEM) << std::endl;
        avformat_free_context(format_context);
        av_dict_free(&options);
        return -1;
    }

    entry = av_dict_get(options, "", nullptr, AV_DICT_IGNORE_SUFFIX);
    if(entry) {
        std::cout << "Dictionary entry after opening input: key = " << entry->key << ", value = " << entry->value << std::endl;
    } else {
        std::cout << "No dictionary entry found after opening input." << std::endl;
    }

    avformat_free_context(format_context);
    av_dict_free(&options);
    return 0;
}

/**
 * yuv的的一些基础相关信息：
 * 1. YUV图片直接存储其二进制数据，没有文件头信息。
 * 2. YUV格式有三大类：
 *   - planner：平面格式，即先连续存储Y分量，然后是U分量，最后是V分量。注意：是先存储整张图片的Y分量，然后是U分量和V分量。
 *   - semi-Planar：半平面格式，即Y分量单独存储，U和V分量交叉存储。注意：是先存储整张图片的Y分量，然后是U和V分量交叉存储。
 *   - packed：打包格式，每个像素点的Y,U,V是连续交错存储的。
 * 3.YUV码流根据不同的采样格式分为不同的格式，如YUV420、YUV422、YUV444等。
 *   - 取样是指按照一定的间隔取值，其中的比例表示Y、U、V分别占的比值。
 *   - YUV444：一个像素点有一个Y分量、一个U分量和一个V分量。即每 4 个 Y 采样，就对应 4 个 Cb 和 4 个 Cr 采样，也就是一个像素占用 8+8+8=24 位
 *     这种存储方式图像质量最高，但空间占用也最大，空间占用与 RGB 存储时一样。
 *   - YUV422: 每 4 个 Y 采样，对应 2 个 Cb 和 2 个 Cr 采样，这样在解析时就会有一些像素点只有亮度信息而没有色度信息，
 *     缺失的色度信息就需要在解析时由相邻的其他色度信息根据一定的算法填充。这种方式下平均一个像素占用空间为 8+4+4=16 位。
 *   - YUV420: 每 4 个 Y 采样，对应 2 个 U 采样或者 2 个 V 采样，注意其中并不是表示 2 个 U 和 0 个 V。
 *     该存储格式下，平均每个像素占用空间为 8+4+0=12 位
 */
void yuv2jpg(std::string yuv_file) {
    std::ifstream yuv_stream(yuv_file, std::ios::binary);
    if(!yuv_stream.is_open()) {
        std::cerr << "Could not open YUV file: " << yuv_file << std::endl;
        return;
    }
    std::cout << "------ yuv file: " << yuv_file << std::endl;
    std::string yuv = "yuv";
    std::string::iterator it = std::find_end(yuv_file.begin(), yuv_file.end(), yuv.begin(), yuv.end());
    if(it == yuv_file.end()) {
        std::cerr << "The file is not a YUV file: " << yuv_file << std::endl;
        return;
    }
    std::string jpg_file = yuv_file.substr(0, it - yuv_file.begin() - 1) + ".jpg";
    std::cout << "------ jpg file: " << jpg_file << std::endl;

    int width = 960;
    int height = 540;
    int frame_size = width * height * 3 / 2; // YUV420格式，每个像素点占用1.5个字节
    
    std::vector<unsigned char> yuv_data(frame_size);
    yuv_stream.read(reinterpret_cast<char*>(yuv_data.data()), frame_size);
    if(yuv_stream.gcount() != frame_size) {
        std::cerr << "Error reading YUV file: " << yuv_file << std::endl;
        return;
    }

    cv::Mat yuv_image(height + height / 2, width, CV_8UC1, yuv_data.data());
    cv::Mat rgb_image;
    cv::cvtColor(yuv_image, rgb_image, cv::COLOR_YUV2BGR_I420); // YUV420格式转换为BGR格式

    cv::imwrite(jpg_file, rgb_image);
}

int main() {
    std::cout << "============================  open_fun ====================== " << std::endl;
    open_fun();
    std::cout << "============================  demuxer_example ====================== " << std::endl;
    demuxer_example();
    return 0;
}
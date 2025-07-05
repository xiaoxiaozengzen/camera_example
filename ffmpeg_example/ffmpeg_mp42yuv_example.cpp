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
#include "libavutil/opt.h"
}

#include <opencv2/opencv.hpp>

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
int decode_fun() {
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

    /**
     * AVClass
     */
    std::cout << "AVClass name: " << format_context->av_class->class_name << std::endl;
    std::cout << "AVClass version: " << format_context->av_class->version << std::endl;

    /**
     * AVOption
     */
    std::cout << "AVOption name: " << format_context->av_class->option->name << std::endl;

    /**
     * AVInputFormat
     */
    std::cout << "AVInputFormat name: " << format_context->iformat->name << std::endl;
    std::cout << "AVInputFormat long_name: " << format_context->iformat->long_name << std::endl;
    std::cout << "AVInputFormat flags: " << format_context->iformat->flags << std::endl;
    if(format_context->iformat->extensions) {
        std::cout << "AVInputFormat extensions: " << format_context->iformat->extensions << std::endl;
    }

    /**
     * @brief AVFormatContext 是 FFmpeg 中的一个结构体，表示多媒体文件的格式上下文。
     */
    std::cout << "AVFormatContex url: " << format_context->url << std::endl;
    std::cout << "AVFormatContex AV_NOPTS_VALUE: " << AV_NOPTS_VALUE << std::endl;  // 表示无法正确获取数据
    std::cout << "AVFormatContex start_time: " << format_context->start_time << std::endl;
    std::cout << "AVFormatContex AV_TIME_BASE: " << AV_TIME_BASE << std::endl;

    /**
     * duration: 视频的持续时间，单位为 AV_TIME_BASE（通常是微秒）。
     * FFmpeg 从 mp4 里面的 duration读取的
     * MP4 标准定义 mvhd 里面的 duration 以最长的流为准，这个 duration 可能是音频流的时长，也可能是视频流的时长。
     */
    std::cout << "AVFormatContex duration: " << format_context->duration / AV_TIME_BASE << " s" << std::endl;
    std::cout << "AVFormatContex nb_streams: " << format_context->nb_streams << std::endl;
    for(std::size_t i = 0; i < format_context->nb_streams; ++i) {
        /**
         * @brief AVStream 是 FFmpeg 中的一个结构体，表示多媒体流（如音频或视频流）。
         */
        AVStream *stream = format_context->streams[i];
        if(stream) {
            std::cout << "AVStream " << i 
                      << ": index: " << stream->index
                      << ", id: " << stream->id
                      << ". time_base: " << stream->time_base.num << "/" << stream->time_base.den
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
    std::cout << "AVFormatContex bit_rate: " << format_context->bit_rate / 1000 << " kbps" << std::endl;
    std::cout << "AVFormatContex iformat_name: " << format_context->iformat->name << std::endl;
    std::cout << "AVFormatContex iformat_long_name: " << format_context->iformat->long_name << std::endl;


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
     * @note codecpar 包含了编解码器的各种参数，如编码格式、分辨率、比特率等。
     * @note 其中codec_type:
     *       - AVMEDIA_TYPE_VIDEO: 表示视频流，值为 0。
     *       - AVMEDIA_TYPE_AUDIO: 表示音频流，值为 1
     *       - AVMEDIA_TYPE_SUBTITLE: 表示字幕流，值为 2。
     *       - AVMEDIA_TYPE_DATA: 表示数据流，值为 3。
     *       - AVMEDIA_TYPE_ATTACHMENT: 表示附件流，值为 4。
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
        std::cerr << "Could not copy codec parameters to context, because: " << AVERROR(ENOMEM) << std::endl;
        avcodec_free_context(&codec_context);
        avformat_close_input(&format_context);
        avformat_free_context(format_context);
        return -1;
    }

    /**
     * AVCodecContext
     */
    std::cout << "AVCodecContext codec_type: " << av_get_media_type_string(codec_context->codec_type) << std::endl;
    std::cout << "AVCodecContext codec_id: " << avcodec_get_name(codec_context->codec_id) << std::endl;
    std::cout << "AVCodecContext codec_tag: " << codec_context->codec_tag << std::endl;
    std::cout << "AVCodecContext bit_rate: " << codec_context->bit_rate / 1000 << " kbps" << std::endl;
    std::cout << "AVCodecContext time_base: " << codec_context->time_base.num << "/" << codec_context->time_base.den << std::endl;
    std::cout << "AVCodecContext framerate: " << codec_context->framerate.num << "/" << codec_context->framerate.den << std::endl;
    std::cout << "AVCodecContext delay: " << codec_context->delay << std::endl;
    std::cout << "AVCodecContext width: " << codec_context->width << std::endl;
    std::cout << "AVCodecContext height: " << codec_context->height << std::endl;
    std::cout << "AVCodecContext coded_width: " << codec_context->coded_width << std::endl;
    std::cout << "AVCodecContext coded_height: " << codec_context->coded_height << std::endl;
    std::cout << "AVCodecContext has_b_frames: " << codec_context->has_b_frames << std::endl;
    std::cout << "AVCodecContext sample_aspect_ratio: " << codec_context->sample_aspect_ratio.num << "/" << codec_context->sample_aspect_ratio.den << std::endl;

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
    std::cout << "AVCodec type: " << av_get_media_type_string(codec->type) << std::endl;
    std::cout << "AVCodec id: " << avcodec_get_name(codec->id) << std::endl;
    std::cout << "AVCodec capabilities: " << codec->capabilities << std::endl;
    std::cout << "AVCodec max lowres: " << codec->max_lowres << std::endl;
    /**
     * struct AVProfile {
     *   int profile;          // 编码器的配置文件
     *   const char *name;    // 编码器配置文件的名称
     * }
     */
    std::cout << "AVCodec profile = " << codec->profiles[0].profile
              << "AVCodec name = " << codec->profiles[0].name
              << std::endl;

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
        std::cerr << "Could not open codec, because: " << AVERROR(ENOMEM) << std::endl;
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
            if(ret == AVERROR_EOF) {
                // 到达文件末尾
                std::cout << "av_read_frame: End of file reached." << std::endl;
                break;
            } else if(ret == AVERROR(EAGAIN)) {
                // 没有更多的包可供读取
                std::cout << "av_read_frame: No more packets to read." << std::endl;
                break;
            }
            // 其他错误
            std::cerr << "av_read_frame: Could not read frame from input file, because: " << AVERROR(ENOMEM) << std::endl;
            break;
        }
        if(packet->stream_index == 1) {
            continue;  // 跳过音频流，只处理视频流
        }
        if(packet_count == 1) {
            std::cout << "------------------Packet count: " << packet_count << std::endl;
            std::cout << "AVPacket pts: " << packet->pts << std::endl;
            std::cout << "AVPacket dts: " << packet->dts << std::endl;
            std::cout << "AVPacket size: " << packet->size << std::endl;
            std::cout << "AVPacket stream index: " << packet->stream_index << std::endl;
            std::cout << "AVPacket flags: " << packet->flags << std::endl;
            std::cout << "AVPacket side_data_elems: " << packet->side_data_elems << std::endl;
            std::cout << "AVPacket duration: " << packet->duration << std::endl;
            std::cout << "AVPacket pos: " << packet->pos << std::endl;
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
            std::cerr << "Could not send packet to decoder, because: " << AVERROR(ENOMEM) << std::endl;
            break;
        }

        // 释放 AVPacket
        av_packet_unref(packet);
        packet_count++;

        while(true) {
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
                    std::cerr << "Could not receive frame from decoder, because: " << AVERROR(ENOMEM) << std::endl;
                    break;
                }
            }
            frame_count++;
            if(frame_count == 1) {
                /**
                 * @brief 打印解码后的帧信息
                 * 
                 * @note data指向解码后picture的planes数据。
                 * @note linesize表示每个平面（plane）的行大小，单位是字节。
                 * @note linesize会进行对齐处理，通常是16/32/64的倍数。因此会出现lensize=1024，但是width=960的情况。
                 *       此时需要将对齐部分的字节进行忽略，即每行的后1024-960=64字节不需要处理。
                 */
                std::cout << "AVFrame data: " << reinterpret_cast<void*>(frame->data[0]) << std::endl;
                std::cout << "AVFrame linesize[0]: " << frame->linesize[0] << std::endl;
                std::cout << "AVFrame linesize[1]: " << frame->linesize[1] << std::endl;
                std::cout << "AVFrame linesize[2]: " << frame->linesize[2] << std::endl;
                std::cout << "AVFrame linesize[3]: " << frame->linesize[3] << std::endl;
                std::cout << "AVFrame width: " << frame->width << std::endl;
                std::cout << "AVFrame height: " << frame->height << std::endl;
                std::cout << "AVFrame nb_samples: " << frame->nb_samples << std::endl;
                std::cout << "AVFrame format: " << av_get_pix_fmt_name((AVPixelFormat)frame->format) << std::endl;
                std::cout << "AVFrame picture_type: " << av_get_picture_type_char(frame->pict_type) << std::endl;
                /**
                 * @brief sample_aspect_ratio 简称 SAR，表示显示的时候单个像素的宽高比。
                 * @note 例如：存储时一张720*576的视频帧，文件中就是720列576行的像素数据。
                 *       但是显示时，如果设置SAR为16:15，那么每个像素的宽度是高度的16/15，也就是扁的
                 *       播放器拿到SAR后，会根据SAR和分辨率来计算实际的显示宽高比(DAR, Display Aspect Ratio)。
                 *       显示时，播放器会将每个像素的宽度和高度进行缩放，以适应显示设备的宽高比。
                 *       则实际显示的时候，视频的宽度会变成 720 * (16/15) = 768，实际高度不变为576。
                 */
                std::cout << "AVFrame sample_aspect_ratio: " 
                          << frame->sample_aspect_ratio.num << "/" 
                          << frame->sample_aspect_ratio.den << std::endl;
                std::cout << "AVFrame pts: " << frame->pts << std::endl;
                std::cout << "AVFrame pkt_dts: " << frame->pkt_dts << std::endl;
                std::cout << "AVFrame quality: " << frame->quality << std::endl;
                /**
                 * @brief repeat_pict 表示帧的重复次数。
                 * @note 在视频编码中，某些帧可能会被重复显示多次，以实现更平滑的播放效果。
                 *       repeat_pict的单位是半帧
                 * @note 对于逐行扫描的视频，repeat_pict的值通常为0
                 *       对于隔行扫描的视频，repeat_pict的值通常为1，表示改帧需要显示1.5帧的时间
                 */
                std::cout << "AVFrame repeat_pict: " << frame->repeat_pict << std::endl;
                std::cout << "AVFrame color_range: " << frame->color_range << std::endl;
                /**
                 * @brief crop_top、crop_bottom、crop_left、crop_right 表示裁剪区域的大小。
                 * @note 表示改帧在显示的时候应从顶部、底部、左侧和右侧裁剪多少像素。
                 */
                std::cout << "AVFrame crop_top: " << frame->crop_top << std::endl;
                std::cout << "AVFrame crop_bottom: " << frame->crop_bottom << std::endl;
                std::cout << "AVFrame crop_left: " << frame->crop_left << std::endl;
                std::cout << "AVFrame crop_right: " << frame->crop_right << std::endl;

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
                    for(int i = 0; i < frame->height; ++i) {
                        yuv_file.write(reinterpret_cast<const char*>(frame->data[0] + i * frame->linesize[0]), frame->width);
                    }
                    for(int i = 0; i < frame->height / 2; ++i) {
                        yuv_file.write(reinterpret_cast<const char*>(frame->data[1] + i * frame->linesize[1]), frame->width / 2);
                    }
                    for(int i = 0; i < frame->height / 2; ++i) {
                        yuv_file.write(reinterpret_cast<const char*>(frame->data[2] + i * frame->linesize[2]), frame->width / 2);
                    }
                }
                yuv_file.close();
                av_frame_unref(frame);
            }
        }
        
    }

    std::cout << "Total packets read: " << packet_count << std::endl;
    std::cout << "Total frames decoded: " << frame_count << std::endl;


    // 释放 AVFrame
    av_frame_free(&frame);
    // 关闭 AVPacket
    av_packet_free(&packet);
    // 关闭解码器
    avcodec_close(codec_context);
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



int main() {
    std::cout << "============================  decode_fun ====================== " << std::endl;
    decode_fun();
    std::cout << "============================  demuxer_example ====================== " << std::endl;
    demuxer_example();

    return 0;
}
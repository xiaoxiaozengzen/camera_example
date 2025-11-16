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
     * @brief 封装格式上下文。多媒体文件的格式上下文。
     * struct AVFormatContext {
     *  const AVClass *av_class;           // 指向 AVClass 结构体的指针，包含有关此结构体的信息
     *  ff_const59 struct AVInputFormat *iformat; // 指向 AVInputFormat 结构体的指针，表示输入格式
     *  ff_const59 struct AVOutputFormat *oformat; // 指向 AVOutputFormat 结构体的指针，表示输出格式
     *  void* priv_data;                  // 指向私有数据的指针，通常用于存储特定格式的上下文信息
     *  struct AVIOContext *pb;            // 指向 AVIOContext 结构体的指针，表示输入/输出的 I/O 上下文
     *  int ctx_flags;                   // 上下文标志
     *  unsigned int nb_streams;        // 流的数量
     *  struct AVStream **streams;      // 指向 AVStream 结构体指针数组的指针，表示媒体流
     *  char* url;                      // 输入/输出的 URL 或文件名
     *  int64_t start_time;             // 媒体的起始时间戳
     *  int64_t duration;               // 媒体的持续时间，单位为 AV_TIME_BASE
     *  int64_t bit_rate;               // 媒体的比特率，单位为 bps（比特每秒）
     *  unsigned int packet_size;      // 数据包的大小
     *  int max_delay;                 // 最大延迟，单位为微秒
     *  int flags;                     // 格式标志，一些列的AVFMT_FLAG_*构成
     *  int64_t probesize;              // 用于探测输入格式的字节数
     *  int64_t max_analyze_duration; // 用于分析输入格式的最大持续时间，单位为微秒
     *  const uint8_t *key;            // 指向密钥数据的指针
     *  int keylen;                   // 密钥数据的长度
     *  unsigned int nb_programs;    // 节目的数量
     *  struct AVProgram **programs;  // 指向 AVProgram 结构体指针数组的指针，表示节目的信息
     *  enum AVCodecID video_codec_id; // 视频编解码器的 ID
     *  enum AVCodecID audio_codec_id; // 音频编解码器的 ID
     *  enum AVCodecID subtitle_codec_id; // 字幕编解码器的 ID
     *  ... // 其他成员省略
     * }
     * 
     * @note AV_TIME_BASE 是 FFmpeg 中用于表示时间基准的常量，值为 1000000，表示以微秒为单位。
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
     * @brief 打开一个输入流并读取其头部信息。此时，codec并未被打开，需要后续再打开。注意：一定要调用avformat_close_input()去关闭输入流.
     * @param ps 指向 AVFormatContext 的指针的地址。这个指针将被填充为打开的格式上下文。
     * @param url 输入流的 URL 或文件名。
     * @param fmt 指定输入格式，强制使用特定的输入格式。如果为 NULL，FFmpeg 将自动检测格式。
     * @param options 指向 AVDictionary 的指针，用于设置打开输入流时的选项。
     * @return 成功：返回0；失败：释放掉ps，并设置其指向null，并返回一个负数AVERROR
     *
     */
    int ret = avformat_open_input(&format_context, mp4_file.c_str(), nullptr, nullptr);
    if(ret < 0) {
        std::cerr << "Could not open input file '" << mp4_file << std::endl;
        return -1;
    }

    /**
     * struct AVClass {
     *  const char *class_name;          // 类的名称
     *  int version;                     // 类的版本
     *  const AVOption *option;          // 选项列表
     *  int log_level_offset_offset;    // 日志级别偏移量
     *  AVClassCategory category;        // 类别
     * }
     */
    std::cout << "AVClass name: " << format_context->av_class->class_name << std::endl;
    std::cout << "AVClass version: " << std::hex << format_context->av_class->version << std::endl;
    std::cout << "AVClass category: " << std::dec << format_context->av_class->category << std::endl;

    /**
     * struct AVOption {
     *  const char *name;             // 选项的名称
     *  const char *help;             // 选项的帮助信息
     *  int offset;                 // 选项在结构体中的偏移量
     *  enum AVOptionType type;    // 选项的类型
     *  union {...} default_val; // 选项的默认值
     *  int flags;                  // 选项的标志
     *  const char* unit;          // 选项的单位
     * }
     */
    std::cout << "AVOption name: " << format_context->av_class->option->name << std::endl;

    /**
     * struct AVInputFormat {
     *  const char *name;                     // 输入格式的名称
     *  const char *long_name;                // 输入格式的长名称
     *  int flags;                          // 输入格式的标志
     *  const char *extensions;               // 支持的文件扩展名列表
     *  const struct AVCodecTag * const *codec_tag; // 支持的编解码器标签
     *  const AVClass *priv_class;          // 指向私有类的指针
     *  const char *mime_type;                // 支持的 MIME 类型
     *  ff_const59 struct AVInputFormat *next; // 指向下一个输入格式的指针
     *  int raw_codec_id;                  // 原始编解码器 ID
     *  ... // 其他成员省略
     * }
     */
    std::cout << "AVInputFormat name: " << format_context->iformat->name << std::endl;
    std::cout << "AVInputFormat long_name: " << format_context->iformat->long_name << std::endl;
    std::cout << "AVInputFormat flags: " << format_context->iformat->flags << std::endl;
    if(format_context->iformat->extensions) {
        std::cout << "AVInputFormat extensions: " << format_context->iformat->extensions << std::endl;
    }
    std::cout << "AVInputFormat raw_codec_id: " << format_context->iformat->raw_codec_id << std::endl;

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

    /**
     * @brief AVIOContext 是 FFmpeg 中的一个结构体，表示输入/输出的 I/O 上下文。
     * struct AVIOContext {
     *  const AVClass *av_class;       // 指向 AVClass 结构体的指针，包含有关此结构体的信息
     *  unsigned char *buffer;        // 指向缓冲区的指针
     *  int buffer_size;             // 缓冲区的大小
     *  unsigned char *buf_ptr;      // 指向当前缓冲区位置的指针
     *  unsigned char *buf_end;      // 指向缓冲区末尾的指针
     *  void *opaque;                // 指向私有数据的指针
     *  int64_t pos;                 // 当前文件位置
     *  int eof_reached;            // 是否到达文件末尾的标志
     *  int write_flag;             // 写入标志
     *  int max_packet_size;        // 最大数据包大小
     *  ... // 其他成员省略
     * }
     */
    std::cout << "AVFormatContext AVIOContext ptr: " << format_context->pb << std::endl;
    std::cout << "AVFormatContext url: " << format_context->url << std::endl;
    if(format_context->start_time != AV_NOPTS_VALUE) {
        std::cout << "AVFormatContext start_time: " << format_context->start_time / AV_TIME_BASE << " s" << std::endl;
    } else {
        std::cout << "AVFormatContext start_time: AV_NOPTS_VALUE = " << AV_NOPTS_VALUE << std::endl;
    }
    std::cout << "AVFormatContext AV_TIME_BASE: " << AV_TIME_BASE << std::endl;
    std::cout << "AVFormatContext duration: " << format_context->duration / AV_TIME_BASE << " s" << std::endl;
    std::cout << "AVFormatContext ctx_flags: " << format_context->ctx_flags << std::endl;
    std::cout << "AVFormatContext nb_streams: " << format_context->nb_streams << std::endl;
    std::cout << "AVFormatContext bit_rate: " << format_context->bit_rate / 1000 << " kbps" << std::endl;
    std::cout << "AVFormatContext iformat_name: " << format_context->iformat->name << std::endl;
    std::cout << "AVFormatContext iformat_long_name: " << format_context->iformat->long_name << std::endl;

    for(std::size_t i = 0; i < format_context->nb_streams; ++i) {
        /**
         * @brief AVStream 是 FFmpeg 中的一个结构体，表示多媒体流（如音频或视频流）。
         * struct AVStream {
         *  int index;                      // 流的索引
         *  int id;                         // 流的 ID
         *  void *priv_data;                // 指向私有数据的指针
         *  AVRational time_base;          // 流的时间基准
         *  int64_t start_time;             // 流的起始时间戳
         *  int64_t duration;               // 流的持续时间
         *  int64_t nb_frames;              // 流中的帧数
         *  int disposition;                // 流的处置标志
         *  enum AVDiscard discard;        // 流的丢弃标志
         *  AVRational sample_aspect_ratio; // 流的采样宽高比
         *  AVDictionary *metadata;        // 指向元数据的指针
         *  AVRational avg_frame_rate;      // 流的平均帧率
         *  AVPacket attached_pic;         // 附加图片（如封面）
         *  ... // 其他成员省略
         *  AVCodecParameters *codecpar; // 指向编解码器参数的指针
         *  ... // 其他成员省略
         * }
         * 
         * struct AVRational {
         *  int num; // 分子
         *  int den; // 分母
         * }
         */
        AVStream *stream = format_context->streams[i];
        if(stream) {
            std::cout << "AVStream " << i 
                      << ": index: " << stream->index
                      << ", id: " << stream->id
                      << ". time_base: " << stream->time_base.num << "/" << stream->time_base.den
                      << ", start_time: " << stream->start_time
                      << ", duration: " << stream->duration
                      << ", duration (s): " << static_cast<double>(stream->duration * stream->time_base.num) / stream->time_base.den
                      << ", nb_frames: " << stream->nb_frames
                      << ", disposition: " << stream->disposition
                      << ", pts_wrap_bits: " << stream->pts_wrap_bits
                      << ", codec_type: " << av_get_media_type_string(stream->codecpar->codec_type)
                      << ", codec_id: " << avcodec_get_name(stream->codecpar->codec_id) 
                      << std::endl;
        }
    }

    /**
     * @brief AVCodecParameters 是 FFmpeg 中的一个结构体，表示编解码器参数。包含了编解码器的各种参数，如编码格式、分辨率、比特率等。
     * struct AVCodecParameters {
     *  enum AVMediaType codec_type;    // 编解码器类型（音频、视频等）
     *  enum AVCodecID codec_id;        // 编解码器 ID
     *  uint32_t codec_tag;         // 编解码器标签
     *  uint8_t* extradata;      // 额外数据
     *  int extradata_size;        // 额外数据的大小
     *  int format;                   // 媒体格式：视频的像素格式或音频的采样格式
     *  int bit_rate;                 // 比特率
     *  int bits_per_coded_sample;   // 每个编码样本的位数
     *  int bits_per_raw_sample;     // 每个原始样本的位数
     *  int profile;                  // 编解码器配置文件
     *  int level;                    // 编解码器级别
     *  int width;                   // 视频宽度
     *  int height;                  // 视频高度
     *  AVRational sample_aspect_ratio; // 采样宽高比
     *  enum AVFieldOrder field_order; // 视频场序
     *  enum AVColorRange color_range; // 颜色范围
     *  enum AVColorPrimaries color_primaries; // 颜色基准
     *  enum AVColorTransferCharacteristic color_trc; // 颜色传输特性
     *  enum AVColorSpace colorspace; // 颜色空间
     *  enum AVChromaLocation chroma_location; // 色度位置
     *  int video_delay;            // 视频延迟
     *  int channel_layout;         // 音频通道布局
     *  int channels;               // 音频通道数
     *  ... // 其他成员省略
     * }
     * 
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
    if(codecpar->extradata && codecpar->extradata_size > 0) {
        std::cout << "AVCodecParameters extradata_size: " << codecpar->extradata_size << std::endl;
        std::cout << "AVCodecParameters extradata: ";
        for(int i = 0; i < std::min(codecpar->extradata_size, 16); ++i) {
            std::cout << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(codecpar->extradata[i]) << " ";
        }
        std::cout << std::dec << std::endl;
        if(codecpar->extradata_size > 16) {
            if(codecpar->extradata[0] == 1) {
                int lengthSizeMinusOne = codecpar->extradata[4] & 0x03;
                int nalu_len_size = lengthSizeMinusOne + 1; // typical 4
                std::cout << "AVCodecParameters  (AVCC format), nalu length size: " << nalu_len_size << std::endl;
            } else {
                std::cout << "AVCodecParameters  (Annex-B format or other)" << std::endl;
            }
        }
    }


    /**
     * @brief AVCodecContext 是 FFmpeg 中的一个结构体，表示编解码器上下文。
     * struct AVCodecContext {
     *  const AVClass *av_class;           // 指向 AVClass 结构体的指针，包含有关此结构体的信息
     *  int log_level_offset;            // 日志级别偏移量
     *  enum AVMediaType codec_type;    // 编解码器类型（音频、视频等）
     *  const struct AVCodec *codec;      // 指向 AVCodec 结构体的指针，表示编解码器
     *  enum AVCodecID codec_id;        // 编解码器 ID
     *  unsigned int codec_tag;         // 编解码器标签
     *  void *priv_data;                // 指向私有数据的指针
     *  struct AVCodecInternal *internal; // 指向 AVCodecInternal 结构体的指针，包含内部数据
     *  void *opaque;                // 指向私有数据的指针
     *  int bit_rate;                   // 比特率
     *  ... // 其他成员省略
     *  int width, height;            // 视频宽度和高度
     *  int coded_width, coded_height; // 编码后的视频宽度和高度
     *  int gop_size;                // 一组pictures中图像的数量
     *  enum AVPixelFormat pix_fmt; // 像素格式
     *  int max_b_frames;           // 最大B帧数
     *  float b_quant_factor;    // B帧量化因子
     *  ... // 其他成员省略
     * }
     */
    AVCodecContext *codec_context = nullptr;
    codec_context = avcodec_alloc_context3(nullptr);
    if(!codec_context) {
        std::cerr << "Could not allocate AVCodecContext, because: " << AVERROR(ENOMEM) << std::endl;
        avformat_close_input(&format_context);
        avformat_free_context(format_context);
        return -1;
    }
    ret = avcodec_parameters_to_context(codec_context, codecpar);
    if(ret < 0) {
        std::cerr << "Could not copy codec parameters to context, because: " << AVERROR(ENOMEM) << std::endl;
        avcodec_free_context(&codec_context);
        avformat_close_input(&format_context);
        avformat_free_context(format_context);
        return -1;
    }
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
     * struct AVCodec {
     *  const char *name;               // 编解码器的名称
     *  const char *long_name;          // 编解码器的长名称
     *  enum AVMediaType type;        // 编解码器类型（音频、视频等）
     *  enum AVCodecID id;            // 编解码器 ID
     *  int capabilities;             // 编解码器的能力标志，参考 AV_CODEC_CAP_* 宏定义
     *  const AVRational *supported_framerates; // 支持的帧率列表
     *  const enum AVPixelFormat *pix_fmts;    // 支持的像素格式列表
     *  const int *supported_samplerates;      // 支持的采样率列表
     *  const enum AVSampleFormat *sample_fmts;   // 支持的采样格式列表
     *  ... // 其他成员省略
     * }
     *
     * struct AVProfile {
     *   int profile;          // 编码器的配置文件
     *   const char *name;    // 编码器配置文件的名称
     * }
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
    std::cout << "AVCodec profile = " << codec->profiles[0].profile
              << ", AVCodec name = " << codec->profiles[0].name
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
     * @brief AVBSFContext 是 FFmpeg 中的一个结构体，表示比特流过滤器上下文。一般用于将NALU的格式从AVCC转换为Annex-B格式。
     * struct AVBSFContext {
     *  const AVClass *av_class;       // 指向 AVClass 结构体的指针，包含有关此结构体的信息
     *  const struct AVBitStreamFilter *filter; // 指向 AVBitStreamFilter 结构体的指针，表示比特流过滤器
     *  AVBSFInternal *internal; // 指向 AVBSFInternal 结构体的指针，包含内部数据
     *  void *priv_data;            // 指向私有数据的指针
     *  AVCodecParameters *par_in;    // 指向输入编解码器参数的指针
     *  AVCodecParameters *par_out;   // 指向输出编解码器
     *  AVRational time_base_in;   // 输入时间基准
     *  AVRational time_base_out;  // 输出时间基准
     * };
     *
     * struuct AVBitStreamFilter {
     *  const char *name;               // 过滤器的名称
     *  const enum AVCodecID *codec_ids; // 支持的编解码器 ID 列表
     *  const AVClass *priv_class;          // 指向私有类的指针
     *  int priv_data_size;          // 私有数据的大小
     * };
     */
    AVBSFContext *bsfc = NULL;
    const AVBitStreamFilter *bsf = av_bsf_get_by_name("h264_mp4toannexb");
    if(!bsf) {
        std::cerr << "Could not find AVBitStreamFilter h264_mp4toannexb" << std::endl;
        av_bsf_free(&bsfc);
        avcodec_free_context(&codec_context);
        avformat_close_input(&format_context);
        avformat_free_context(format_context);
        return -1;
    }
    ret = av_bsf_alloc(bsf, &bsfc);
    if(ret < 0) {
        std::cerr << "Could not allocate AVBSFContext, because: " << AVERROR(ENOMEM) << std::endl;
        av_bsf_free(&bsfc);
        avcodec_free_context(&codec_context);
        avformat_close_input(&format_context);
        avformat_free_context(format_context);
        return -1;
    }
    avcodec_parameters_copy(bsfc->par_in, codecpar);
    ret = av_bsf_init(bsfc);
    if(ret < 0) {
        std::cerr << "Could not initialize AVBSFContext, because: " << AVERROR(ENOMEM) << std::endl;
        av_bsf_free(&bsfc);
        avcodec_free_context(&codec_context);
        avformat_close_input(&format_context);
        avformat_free_context(format_context);
        return -1;
    }
    std::cout << "AVBFSContext AVCodecParameters codec_type: " << av_get_media_type_string(bsfc->par_in->codec_type) << std::endl;
    std::cout << "AVBFSContext AVCodecParameters codec_id: " << avcodec_get_name(bsfc->par_in->codec_id) << std::endl;
    std::cout << "AVBFSContext AVCodecParameters codec_tag: " << bsfc->par_in->codec_tag << std::endl;
    std::cout << "AVBFSContext AVCodecParameters format: " << bsfc->par_in->format << std::endl;
    std::cout << "AVBFSContext AVCodecParameters bit_rate: " << bsfc->par_in->bit_rate / 1000 << " kbps" << std::endl;
    std::cout << "AVBFSContext AVCodecParameters profile: " << bsfc->par_in->profile << std::endl;
    std::cout << "AVBFSContext AVCodecParameters level: " << bsfc->par_in->level << std::endl;
    std::cout << "AVBFSContext AVCodecParameters width: " << bsfc->par_in->width << std::endl;
    std::cout << "AVBFSContext AVCodecParameters height: " << bsfc->par_in->height << std::endl;

    /**
     * @brief AVPacket 是 FFmpeg 中的一个结构体，表示多媒体数据包。保存编码后的压缩数据，通常是音频或视频帧。
     * struct AVPacket {
     *  AVBufferRef *buf;          // 指向 AVBufferRef 结构体的指针，表示数据缓冲区。可以存储packet的data
     *  int64_t pts;               // 显示时间戳（Presentation Time Stamp）,单位是AVStream->time_base
     *  int64_t dts;               // 解码时间戳（Decoding Time Stamp）,单位是AVStream->time_base, 送给解码器的时间戳，这个可能跟pts不一样，因为B帧会导致重新排序
     *  uint8_t *data;            // 指向数据的指针
     *  int size;                  // 数据的大小
     *  int stream_index;         // 数据所属的流的索引
     *  int flags;                 // 数据包的标志
     *  AVPacketSideData *side_data; // 指向 AVPacketSideData 结构体的指针，表示附加数据
     *  int side_data_elems;      // 附加数据的数量
     *  int duration;              // 数据包的持续时间，单位是AVStream->time_base
     *  int64_t pos;               // 数据包在stream中的位置
     * }
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

    AVPacket *filtered_packet = av_packet_alloc();
    if(!filtered_packet) {
        std::cerr << "Could not allocate filtered AVPacket, because: " << AVERROR(ENOMEM) << std::endl;
        av_packet_free(&packet);
        avcodec_free_context(&codec_context);
        avformat_close_input(&format_context);
        avformat_free_context(format_context);
        return -1;
    }

    /**
     * @brief AVFrame 是 FFmpeg 中的一个结构体，表示解码后的帧数据。
     * struct AVFrame {
     *  uint8_t *data[AV_NUM_DATA_POINTERS]; // 指向各个平面（plane）数据的指针数组
     *  int linesize[AV_NUM_DATA_POINTERS]; // 各个平面的行大小，单位是字节
     *  uint8_t **extended_data;            // 指向扩展数据的指针数组
     *  int width, height;                  // 视频宽度和高度
     *  int nb_samples;                   // 音频样本数
     *  int format;                         // 媒体格式：视频的像素格式或音频的采样格式
     *  int key_frame;                     // 是否为关键帧
     *  enum AVPictureType pict_type;     // 视频帧的类型
     *  AVRational sample_aspect_ratio; // 采样宽高比
     *  int64_t pts;                       // 显示时间戳（Presentation Time Stamp）,单位是AVStream->time_base
     *  int64_t pkt_pts;                   // 数据包的显示时间戳
     *  int coded_picture_number;      // 图片在编码序列中的编号
     *  int display_picture_number;    // 图片在显示序列中的编号
     *  int quality;                       // 图片质量
     *  ... // 其他成员省略
     * }
     *
     * @note data指向解码后picture的planes数据。
     * @note linesize表示每个平面（plane）的行大小，单位是字节。
     *       linesize会进行对齐处理，通常是16/32/64的倍数。因此会出现lensize=1024，但是width=960的情况。
     *       此时需要将对齐部分的字节进行忽略，即每行的后1024-960=64字节不需要处理。
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
        if(packet_count == 1 || packet_count == 30) {
            std::cout << "------------------Packet count: " << packet_count << std::endl;
            std::cout << "AVPacket pts: " << packet->pts << ", s = " << packet->pts / AV_TIME_BASE << std::endl;
            std::cout << "AVPacket dts: " << packet->dts << ", s = " << packet->dts / AV_TIME_BASE << std::endl;
            std::cout << "AVPacket data ptr: " << reinterpret_cast<void*>(packet->data) << std::endl;
            std::cout << "AVPacket size: " << packet->size << std::endl;
            std::cout << "AVPacket stream index: " << packet->stream_index << std::endl;
            std::cout << "AVPacket flags: " << packet->flags << std::endl;
            std::cout << "AVPacket side_data_elems: " << packet->side_data_elems << std::endl;
            std::cout << "AVPacket duration: " << packet->duration << ", s = " << packet->duration / AV_TIME_BASE << std::endl;
            std::cout << "AVPacket pos: " << packet->pos << std::endl;
            std::cout << "AVPacket data (first 8 bytes): ";
            for(int i = 0; i < 8 && i < packet->size; ++i) {
                std::cout << static_cast<int>(packet->data[i]) << " ";
            }
            std::cout << std::endl;
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

        ret = av_bsf_send_packet(bsfc, packet);
        if(ret < 0) {
            std::cerr << "Could not send packet to AVBSFContext, because: " << AVERROR(ENOMEM) << std::endl;
            break;
        }

        // 释放packet中的数据，但是packet结构体还可以继续使用，避免重复分配内存
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
            if(frame_count == 1 || frame_count == 30) {
                std::cout << "AVFrame data: " << reinterpret_cast<void*>(frame->data[0]) << std::endl;
                std::cout << "AVFrame linesize[0]: " << frame->linesize[0] << std::endl;
                std::cout << "AVFrame linesize[1]: " << frame->linesize[1] << std::endl;
                std::cout << "AVFrame linesize[2]: " << frame->linesize[2] << std::endl;
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
                 * struct AVComponentDescriptor {
                 *  int plane;   // 该组件所在的平面
                 *  int step;    // 组件之间的步长，以字节为单位
                 *  int offset;  // 组件在像素中的偏移量，以字节为单位
                 *  int shift;   // 组件的位移量
                 *  int depth;   // 组件的位深度，以位为单位
                 * }
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
                AVComponentDescriptor comp_v = pix_fmt_desc->comp[2];
                std::cout << "AVComponentDescriptor plane_v: " << (uint16_t)comp_v.plane << std::endl;
                std::cout << "AVComponentDescriptor step_v: " << comp_v.step << std::endl;
                std::cout << "AVComponentDescriptor depth_v: " << (uint16_t)comp_v.depth << std::endl;
                std::cout << "AVComponentDescriptor offset_v: " << (uint16_t)comp_v.offset << std::endl;
                std::cout << "AVComponentDescriptor shift_v: " << (uint16_t)comp_v.shift << std::endl;
            }

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

        while(true) {
            ret = av_bsf_receive_packet(bsfc, filtered_packet);
            if(ret < 0) {
                if(ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                    // EAGAIN 表示没有更多的包可供读取，EOF 表示到达了文件末尾
                    break;
                } else {
                    std::cerr << "Could not receive packet from AVBSFContext, because: " << AVERROR(ENOMEM) << std::endl;
                    break;
                }
            }
            if(packet_count == 1 || packet_count == 30) {
                std::cout << "Filtered AVPacket pts: " << filtered_packet->pts << ", s = " << filtered_packet->pts / AV_TIME_BASE << std::endl;
                std::cout << "Filtered AVPacket dts: " << filtered_packet->dts << ", s = " << filtered_packet->dts / AV_TIME_BASE << std::endl;
                std::cout << "Filtered AVPacket data ptr: " << reinterpret_cast<void*>(filtered_packet->data) << std::endl;
                std::cout << "Filtered AVPacket size: " << filtered_packet->size << std::endl;
                std::cout << "Filtered AVPacket stream index: " << filtered_packet->stream_index << std::endl;
                std::cout << "Filtered AVPacket flags: " << filtered_packet->flags << std::endl;
                std::cout << "Filtered AVPacket side_data_elems: " << filtered_packet->side_data_elems << std::endl;
                std::cout << "Filtered AVPacket duration: " << filtered_packet->duration << ", s = " << filtered_packet->duration / AV_TIME_BASE << std::endl;
                std::cout << "Filtered AVPacket pos: " << filtered_packet->pos << std::endl;
                std::cout << "Filtered AVPacket data (first 8 bytes): ";
                for(int i = 0; i < 8 && i < filtered_packet->size; ++i) {
                    std::cout << static_cast<int>(filtered_packet->data[i]) << " ";
                }
                std::cout << std::endl;
            }
            
            // 释放filtered_packet中的数据，但是filtered_packet结构体还可以继续使用，避免重复分配内存
            av_packet_unref(filtered_packet);
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
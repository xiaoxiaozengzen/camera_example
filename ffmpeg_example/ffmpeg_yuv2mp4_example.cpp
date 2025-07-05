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
 * yuv转换为mp4的流程：
 * 1.avformat_alloc_output_context2：创建一个输出格式上下文，用于封装 MP4 文件。
 * 2.avformat_new_stream：在格式上下文中添加一个视频流。
 * 3.avcodec_find_encoder：查找 H.264 编码器。
 * 4.avcodec_alloc_context3：基于查找到的编码器，为编码器分配一个上下文。
 * 5.avcodec_parameters_from_context：将编码器上下文的参数复制到视频流的编解码参数中。
 * 6.avcodec_open2：将编码器跟编码器上下文关联起来，准备编码。
 * 7.avio_open2：打开输出文件的 I/O 上下文，以便写入 MP4 文件。
 * 8.avformat_write_header：写入 MP4 文件的头部信息。
 * 9.循环操作：
 *   - 填充 AVFrame 的数据，通常是从 YUV 图像文件中读取数据。
 *   - av_frame_get_buffer：为 AVFrame 分配内存缓冲区，以存储编码后的数据。
 *   - avcodec_send_frame：将 AVFrame 发送到编码器进行编码。
 *   - avcodec_receive_packet：从编码器接收编码后的数据包。
 *   - av_packet_rescale_ts：将数据包的时间戳从编码器上下文的时间基准转换为视频流的时间基准。
 *   - av_interleaved_write_frame：将编码后的数据包写入 MP4 文件。
 *   - av_packet_unref：释放 AVPacket，准备下一次编码。
 * 10.循环结束后，调用 av_write_trailer 写入 MP4 文件的尾部信息。
 * 11.释放资源：关闭编码器、释放 AVFrame 和 AVPacket、关闭输出文件的 I/O 上下文、释放格式上下文等。
 */
int encode_fun() {
    int ret = 0;

    std::string mp4_file = "/mnt/workspace/cgz_workspace/Exercise/camera_example/output/generate.mp4";

    /**
     * 1.封装格式上下文。可以理解为 MP4 文件的容器格式。
     */
    AVFormatContext *format_context = nullptr;
    ret = avformat_alloc_output_context2(&format_context, nullptr, NULL, mp4_file.c_str());
    if(ret < 0 || !format_context) {
        std::cerr << "Could not allocate output format context, because: " << AVERROR(ENOMEM) << std::endl;
        return -1;
    }

    /**
     * 2.添加一个视频流到格式上下文中。
     */
    AVStream *video_stream = avformat_new_stream(format_context, nullptr);
    video_stream->time_base = {1, 24}; // 设置时间基准为1/24秒

    /**
     * 3.创建编码器的上下文。
     */
    AVCodecContext *codec_context = nullptr;

    /**
     * 4.查找H.264编码器。
     */
    AVCodec* encode = avcodec_find_encoder(AV_CODEC_ID_H264);
    codec_context = avcodec_alloc_context3(encode);
    codec_context->codec_type = AVMEDIA_TYPE_VIDEO;
    codec_context->bit_rate = 4000000; // 4Mbps
    codec_context->framerate = {24, 1}; // 设置帧率为24fps
    codec_context->gop_size = 5; // 用于设置视频编码时，两个关键帧(I帧)之间的帧数
    codec_context->max_b_frames = 2; // 设置连续B帧的最大数量
    /**
     * @brief 视频编码的压缩特性级别
     * 
     * @note 当前有：
     *       - FF_PROFILE_H264_BASELINE: 基线配置，适用于低复杂度的应用，如移动设备。
     *       - FF_PROFILE_H264_MAIN: 主配置，适用于大多数应用，提供更好的压缩效率。
     *       - FF_PROFILE_H264_HIGH: 高配置，适用于高质量视频编码和广播应用。
     *       - FF_PROFILE_H264_HIGH_10: 高10位配置，适用于高动态范围视频编码。
     *       - FF_PROFILE_H264_HIGH_422: 高422配置，适用于高质量视频编码，支持4:2:2色彩采样。
     *       - FF_PROFILE_H264_HIGH_444_PREDICTIVE: 高444预测配置，适用于高质量视频编码，支持4:4:4色彩采样。
     *       - FF_PROFILE_H264_CAVLC_444: CAVLC 444配置，适用于高质量视频编码，支持4:4:4色彩采样。
     *       - FF_PROFILE_H264_SCALABLE_BASELINE: 可伸缩基线配置，适用于可伸缩视频编码。
     *       - FF_PROFILE_H264_SCALABLE_HIGH: 可伸缩高配置，适用于可伸缩视频编码。
     */
    codec_context->profile = FF_PROFILE_H264_HIGH;
    codec_context->time_base = {1, 24};
    codec_context->width = 960; // 视频宽度
    codec_context->height = 540; // 视频高度
    codec_context->pix_fmt = AV_PIX_FMT_YUV420P; // 设置像素格式为YUV420P
    codec_context->sample_aspect_ratio = {1, 1}; // 设置采样宽高比
    /**
     * @brief 色彩范围
     * @note 色彩范围用于指定视频的色彩空间和色彩范围。
     *       - AVCOL_RANGE_UNSPECIFIED: 未指定色彩范围。
     *       - AVCOL_RANGE_MPEG: MPEG色彩范围，YUV范围为16-235。U/V范围为16-240。适用于大多数视频编码(如H.264、Mpeg)。
     *       - AVCOL_RANGE_JPEG: JPEG色彩范围，YUV范围为0-255。U/V范围为0-255。适用于JPEG图像编码。
     */
    codec_context->color_range = AVCOL_RANGE_MPEG; // 设置色彩范围为MPEG
    /**
     * @brief 色彩原色，用于指定视频的色彩原色，即定义红色、绿色和蓝色的原色坐标。
     * @note 色彩原色用于指定视频的色彩空间和色彩原色。
     *       - AVCOL_PRI_UNSPECIFIED: 未指定色彩原色。
     *       - AVCOL_PRI_BT709: BT.709色彩原色，适用于高清电视和大多数视频编码(如H.264、Mpeg)。
     *       - AVCOL_PRI_BT470M: BT.470M色彩原色，适用于旧式电视和视频编码。
     *       - AVCOL_PRI_BT470BG: BT.470BG色彩原色，适用于旧式电视和视频编码。
     *       - AVCOL_PRI_SMPTE170M: SMPTE 170M色彩原色，适用于旧式电视和视频编码
     */
    codec_context->color_primaries = AVCOL_PRI_BT709; // 设置色彩原色为BT.709
    codec_context->color_trc = AVCOL_TRC_BT709; // 设置色彩传递特性为BT.709
    codec_context->colorspace = AVCOL_SPC_BT709; // 设置色彩空间为BT.709
    /**
     * @brief 色度采样位置，用于指定色度信息在视频帧中的位置。主要影响420、422、444等色度采样格式。
     * @note 色度采样位置用于指定色度信息在视频帧中的位置。
     *       - AVCHROMA_LOC_UNSPECIFIED: 未指定色度采样位置。
     *       - AVCHROMA_LOC_LEFT: 色度信息位于左侧。
     *       - AVCHROMA_LOC_CENTER: 色度信息位于中心。
     *       - AVCHROMA_LOC_TOPLEFT: 色度信息位于左上角。
     *       - AVCHROMA_LOC_TOP: 色度信息位于顶部。
     *       - AVCHROMA_LOC_BOTTOMLEFT: 色度信息位于左下角。
     *       - AVCHROMA_LOC_BOTTOM: 色度信息位于底部.
     *       - AVCHROMA_LOC_TOPRIGHT: 色度信息位于右上角。
     *       - AVCHROMA_LOC_BOTTOMRIGHT: 色度信息位于右下角。
     */
    codec_context->chroma_sample_location = AVCHROMA_LOC_LEFT; // 设置色度采样位置为左侧
    /**
     * @brief 帧顺序，用于指定视频帧的扫描方式。
     * @note 帧顺序用于指定视频帧的扫描方式。
     *       - AV_FIELD_UNKNOWN: 未知帧顺序。
     *       - AV_FIELD_PROGRESSIVE: 逐行扫描，即每一帧都是完整的图像。适合高清视频和大多数现代视频编码。
     *       - AV_FIELD_TT，AV_FIELD_BB: 交错扫描，即每一帧由两半组成，分别是奇数行和偶数行。适合旧式电视和视频编码。
     */
    codec_context->field_order = AV_FIELD_PROGRESSIVE; // 设置帧顺序为逐行扫描

    /**
     * 5.将编解码参数从解码器上下文复制到流的编解码参数。
     */
    ret = avcodec_parameters_from_context(video_stream->codecpar, codec_context);
    if(ret < 0) {
        std::cerr << "Could not copy codec parameters to stream, because: " << AVERROR(ENOMEM) << std::endl;
        avcodec_free_context(&codec_context);
        avformat_free_context(format_context);
        return -1;
    }

    /**
     * 6.将编码器上下文与编码器关联。
     */
    ret = avcodec_open2(codec_context, encode, nullptr);
    if(ret < 0) {
        std::cerr << "Could not open codec, because: " << AVERROR(ENOMEM) << std::endl;
        avcodec_free_context(&codec_context);
        return -1;
    }

    /**
     * 7.正式打开输出文件。
     */
    ret = avio_open2(&format_context->pb, mp4_file.c_str(), AVIO_FLAG_WRITE, &format_context->interrupt_callback, nullptr);
    if(ret < 0) {
        std::cerr << "Could not open output file '" << mp4_file << "', because: " << AVERROR(ENOMEM) << std::endl;
        avcodec_free_context(&codec_context);
        avformat_free_context(format_context);
        return -1;
    }

    /**
     * 8.写入输出文件的头部信息。
     */
    ret = avformat_write_header(format_context, nullptr);
    if(ret < 0) {
        std::cerr << "Could not write header to output file, because: " << AVERROR(ENOMEM) << std::endl;
        avio_closep(&format_context->pb);
        avcodec_free_context(&codec_context);
        avformat_free_context(format_context);
        return -1;
    }

    AVPacket *packet = av_packet_alloc();
    if(!packet) {
        std::cerr << "Could not allocate AVPacket, because: " << AVERROR(ENOMEM) << std::endl;
        avcodec_free_context(&codec_context);
        return -1;
    }
    AVFrame *frame = av_frame_alloc();
    if(!frame) {
        std::cerr << "Could not allocate AVFrame, because: " << AVERROR(ENOMEM) << std::endl;
        av_packet_free(&packet);
        avcodec_free_context(&codec_context);
        return -1;
    }

    std::string image_path = "/mnt/workspace/cgz_workspace/Exercise/camera_example/output";
    std::uint32_t frame_count = 0;
    std::uint32_t packet_count = 0;
    while(true) {
        std::stringstream image_name;
        image_name << std::setw(3) << std::setfill('0') << frame_count;
        std::string image_file = image_path + "/frame_" + image_name.str() + ".yuv";
        std::cout << "=================== Processing image: " << image_file << std::endl;
        std::ifstream yuv_file(image_file, std::ios::binary);
        if(!yuv_file.is_open()) {
            std::cerr << "Could not open input file '" << image_file << "' for reading." << std::endl;
            break;
        }

        frame->format = codec_context->pix_fmt;
        frame->width = codec_context->width;
        frame->height = codec_context->height;
        /**
         * @brief 设置AVFrame的pts(时间戳)
         * @note 必须严格的单调递增
         */
        frame->pts = frame_count;

        /**
         * @brief 分配AVFrame的实际数据缓冲区
         * @note 在调用该函数之前，需要先设置frame的format、width和height等属性。
         *       该函数会根据这些属性分配实际的数据缓冲区，并将frame->data和frame->linesize指向这些缓冲区。
         * @param frame 要分配数据缓冲区的AVFrame指针。
         * @param align 对齐方式，一般建议设置为0，让系统自动选择合适的对齐方式。
         * @return 返回0表示成功，返回负值表示失败。
         */
        ret = av_frame_get_buffer(frame, 0); // 分配实际数据缓冲区
        if(ret < 0) {
            std::cerr << "Could not allocate frame data buffer, because: " << AVERROR(ENOMEM) << std::endl;
            yuv_file.close();
            av_packet_free(&packet);
            av_frame_free(&frame);
            avcodec_free_context(&codec_context);
            avformat_free_context(format_context);
            return -1;
        }

        int y_size = frame->width * frame->height;
        int uv_size = y_size / 4; // UV分量的大小为Y
        yuv_file.read(reinterpret_cast<char*>(frame->data[0]), y_size); // 读取Y分量
        yuv_file.read(reinterpret_cast<char*>(frame->data[1]), uv_size); // 读取U分量
        yuv_file.read(reinterpret_cast<char*>(frame->data[2]), uv_size); // 读取V分量
        yuv_file.close();
        
        avcodec_send_frame(codec_context, frame);
        av_frame_unref(frame);

        while(true) {
            ret = avcodec_receive_packet(codec_context, packet);
            if(ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                break; // 没有更多数据包可接收
            } else if(ret < 0) {
                std::cerr << "Error receiving packet: " << AVERROR(ENOMEM) << std::endl;
                av_packet_free(&packet);
                av_frame_free(&frame);
                avcodec_free_context(&codec_context);
                avformat_free_context(format_context);
                return -1;
            }
            packet_count++;
            std::cout << "packet_count: " << packet_count << std::endl;

            packet->stream_index = video_stream->index; // 设置数据包的流索引

            /**
             * @brief 将AVPacket中的时间戳pts和dts从编码器的时间基准转换为流的时间基准。
             * @note 不同的上下文的时基可能不一样，入编码器和输出流的time_base可能不一样。
             * @note 写入文件时，需要爸packet的时间戳从编码器的时基转换为流的时基。保证播放器能正确解码和播放视频。
             * 
             * @note 必须在调用av_interleaved_write_frame()之前调用av_packet_rescale_ts()函数，
             *       否则写入的时间戳可能不正确，导致视频播放时出现问题。
             *       例如：如果编码器的时间基准是1/4秒，而流的时间基准是1/24秒，
             *       那么需要将编码器的时间戳从1/4秒转换为1/24秒，
             *       否则播放器可能会将视频播放得过快或过慢。
             */
            av_packet_rescale_ts(packet, codec_context->time_base, video_stream->time_base); // 重采样时间戳
            
            /**
             * @brief 将编码后的AVPacket写入输出文件。并自动处理多路流
             */
            ret = av_interleaved_write_frame(format_context, packet);
            if(ret < 0) {
                std::cerr << "Error writing packet to output file, because: " << AVERROR(ENOMEM) << std::endl;
                av_packet_free(&packet);
                av_frame_free(&frame);
                avcodec_free_context(&codec_context);
                avformat_free_context(format_context);
                return -1;
            }
            
            av_packet_unref(packet); // 释放数据包
        }
        
        frame_count++;
    }
    

    av_packet_free(&packet);
    av_frame_free(&frame);
    avcodec_close(codec_context);
    avcodec_free_context(&codec_context);

    // 必须调用av_write_trailer()来写入文件尾部信息。要不然可能写入的文件不完整。
    av_write_trailer(format_context);
    
    avio_closep(&format_context->pb);
    avformat_free_context(format_context);

    return 0;
}

int main() {
    std::cout << "============================  encode_fun ====================== " << std::endl;
    encode_fun();

    return 0;
}
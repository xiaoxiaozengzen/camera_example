#include <iostream>
#include <string>
#include <fstream>
#include <algorithm>
#include <cerrno>
#include <iomanip>
#include <thread>
#include <chrono>

extern "C" {
#include "libavformat/avformat.h"
#include "libavcodec/avcodec.h"
#include "libavutil/pixfmt.h"
#include "libavutil/avutil.h"
#include "libavutil/pixdesc.h"
#include "libavutil/opt.h"
#include <libavcodec/bsf.h>
}

#include <opencv2/opencv.hpp>

class BSFExample {
 public:
    BSFExample(std::string mp4_file, std::string output_h264_mp4)
    : output_h264_mp4_(output_h264_mp4),
      input_mp4_(mp4_file) {

    }
    
    ~BSFExample() {
        if(decode_thread_.joinable()) {
            decode_thread_.join();
        }

        is_inited_ = false;

        // 释放 AVPacket
        av_packet_free(&filtered_packet);
        av_packet_free(&packet);
        // 关闭 BSF 上下文
        if(bsf_context) {
            av_bsf_free(&bsf_context);
        }
        // 关闭解码器
        avcodec_close(codec_context);
        // 关闭解码器
        avcodec_free_context(&codec_context);
        // 关闭输入文件
        avformat_close_input(&format_context);
        // 释放格式上下文
        avformat_free_context(format_context);

        // 关闭输出编码器
        avcodec_close(output_codec_context);
        avcodec_free_context(&output_codec_context);
        av_write_trailer(output_format_context);
        avio_closep(&output_format_context->pb);
        avformat_free_context(output_format_context);
    }

    bool init() {
        avformat_network_init();

        format_context = avformat_alloc_context();
        if(!format_context) {
            std::cerr << "Could not allocate AVFormatContext, because: " << AVERROR(ENOMEM) << std::endl;
            return false;
        } else {
            std::cout << "AVFormatContext allocated successfully." << std::endl;
        }

        int ret = avformat_open_input(&format_context, input_mp4_.c_str(), nullptr, nullptr);
        if(ret < 0) {
            std::cerr << "Could not open input file: " << input_mp4_ << ", error: " << ret << std::endl;
            return false;
        } else {
            std::cout << "Input file opened successfully: " << input_mp4_ << std::endl;
        }

        /**
         * @brief 查找流信息
         * @param ic 指向 AVFormatContext 的指针，表示输入格式上下文。
         * @param options 指向 AVDictionary 的指针，用于设置查找流信息时的选项。可以为 NULL。
         * @return 成功时返回非负值，失败时返回负值错误代码
         */
        ret = avformat_find_stream_info(format_context, nullptr);
        if(ret < 0) {
            std::cerr << "Could not find stream info in input file: " << input_mp4_ << ", error: " << ret << std::endl;
            return false;
        } else {
            std::cout << "Stream info found successfully in input file." << std::endl;
        }
        
        /**
         * @brief 查找最佳视频流
         * @param ic 指向 AVFormatContext 的指针，表示输入格式上下文。
         * @param type 媒体类型，这里使用 AVMEDIA_TYPE_VIDEO 表示视频流。
         * @param wanted_stream_nb 指定想要查找的流的索引，如果为 -1 则表示查找所有流。
         * @param related_stream 指定与所查找流相关的流的索引，如果为 -1 则表示不考虑相关流。
         * @param decoder_ret 指向 AVCodec 的指针的地址，用于返回找到的解码器，如果不需要可以传入 NULL。
         * @param flags 查找标志，通常为 0。
         * @return 成功时返回最佳流的索引，失败时返回负值错误代码。
         */
        int iVideoStream = av_find_best_stream(format_context, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
        if(iVideoStream < 0) {
            std::cerr << "Could not find video stream in input file: " << input_mp4_ << std::endl;
            return false;
        } else {
            std::cout << "Video stream index found: " << iVideoStream << std::endl;
        }
        stream = format_context->streams[iVideoStream];
        if(!stream) {
            std::cerr << "Could not find video stream in input file: " << input_mp4_ << std::endl;
            return false;
        } else {
            std::cout << "Video stream found in input file." << std::endl;
        }

        codec_context = avcodec_alloc_context3(nullptr);
        if(!codec_context) {
            std::cerr << "Could not allocate AVCodecContext, because: " << AVERROR(ENOMEM) << std::endl;
            return false;
        } else {
            std::cout << "AVCodecContext allocated successfully." << std::endl;
        }

        ret = avcodec_parameters_to_context(codec_context, stream->codecpar);
        if(ret < 0) {
            std::cerr << "Could not copy codec parameters to codec context, because: " << ret << std::endl;
            return false;
        } else {
            std::cout << "Codec parameters copied to codec context successfully." << std::endl;
        }

        codec = avcodec_find_decoder(codec_context->codec_id);
        if(!codec) {
            std::cerr << "Could not find decoder for codec_id: " << avcodec_get_name(codec_context->codec_id) << std::endl;
            return false;
        } else {
            std::cout << "Decoder found: " << codec->name << std::endl;
        }

        ret = avcodec_open2(codec_context, codec, nullptr);
        if(ret < 0) {
            std::cerr << "Could not open codec, because: " << AVERROR(ENOMEM) << std::endl;
            return false;
        } else {
            std::cout << "Codec opened successfully." << std::endl;
        }

        packet = av_packet_alloc();
        filtered_packet = av_packet_alloc();
        if(!packet || !filtered_packet) {
            std::cerr << "Could not allocate AVPacket, because: " << AVERROR(ENOMEM) << std::endl;
            return false;
        } else {
            std::cout << "AVPacket allocated successfully." << std::endl;
        }

        const AVBitStreamFilter *bsf = av_bsf_get_by_name("h264_mp4toannexb");
        if(!bsf) {
            std::cerr << "Could not find h264_mp4toannexb bitstream filter" << std::endl;
            return false;
        } else {
            std::cout << "h264_mp4toannexb bitstream filter found." << std::endl;
        }

        /**
         * @brief 基于给定的bitstream filter分配context
         * @param filter AVBitStreamFilter*对象
         * @param ctx AVBSFContext**，需要调用av_bsf_free()
         * @return 0表示成功，失败返回负值
         * @note 后续调用av_bsf_init
         */
        ret = av_bsf_alloc(bsf, &bsf_context);
        if(ret < 0) {
            std::cerr << "Could not allocate BSF context, because: " << AVERROR(ENOMEM) << std::endl;
            return false;
        } else {
            std::cout << "BSF context allocated successfully." << std::endl;
        }
        avcodec_parameters_copy(bsf_context->par_in, stream->codecpar);
        ret = av_bsf_init(bsf_context);
        if(ret < 0) {
            std::cerr << "Could not initialize BSF context, because: " << AVERROR(ENOMEM) << std::endl;
            return false;
        } else {
            std::cout << "BSF context initialized successfully." << std::endl;
        }

        bool output_inited = init_output();
        if(!output_inited) {
            return false;
        }

        is_inited_ = true;
        decode_thread_ = std::thread(&BSFExample::work, this);

        return true;
    }

    bool init_output() {
        int ret = avformat_alloc_output_context2(&output_format_context, nullptr, nullptr, output_h264_mp4_.c_str());
        if(ret < 0 || !output_format_context) {
            std::cerr << "Could not allocate output AVFormatContext, because: " << AVERROR(ENOMEM) << std::endl;
            return false;
        } else {
            std::cout << "Output AVFormatContext allocated successfully." << std::endl;
        }

        output_stream = avformat_new_stream(output_format_context, nullptr);
        if(!output_stream) {
            std::cerr << "Could not create new stream in output format context, because: " << AVERROR(ENOMEM) << std::endl;
            return false;
        } else {    
            std::cout << "New stream created in output format context successfully." << std::endl;
        }
        output_stream->time_base = stream->time_base;
        ret = avcodec_parameters_copy(output_stream->codecpar, stream->codecpar);
        if(ret < 0) {
            std::cerr << "Could not copy codec parameters to output stream, because: " << AVERROR(ENOMEM) << std::endl;
            return false;
        } else {
            std::cout << "Codec parameters copied to output stream successfully." << std::endl;
        }

        ret = avio_open2(&output_format_context->pb, output_h264_mp4_.c_str(), AVIO_FLAG_WRITE, &output_format_context->interrupt_callback, nullptr);
        if(ret < 0) {
            std::cerr << "Could not open output file '" << output_h264_mp4_ << "', because: " << AVERROR(ENOMEM) << std::endl;
            return false;
        } else {
            std::cout << "Output file opened successfully: " << output_h264_mp4_ << std::endl;
        }
        ret = avformat_write_header(output_format_context, nullptr);
        if(ret < 0) {
            std::cerr << "Could not write header to output file, because: " << AVERROR(ENOMEM) << std::endl;
            return false;
        } else {
            std::cout << "Header written to output file successfully." << std::endl;
        }


        return true;
    }

    void work() {
        if(!is_inited_) {
            std::cerr << "BSFExample is not inited!" << std::endl;
            return;
        }
        std::cout << "BSFExample work thread started." << std::endl;

        int packet_count = 0;
        int filtered_packet_count = 0;
        int ret = 0;
        while(true) {
            if(packet->data) {
                av_packet_unref(packet);
            }

            ret = av_read_frame(format_context, packet);
            if(ret < 0) {
                std::cerr << "Could not read frame from input file, error: " << ret << std::endl;
                break;
            }

            if(packet->stream_index != stream->index) {
                continue;
            }

            packet_count++;
            std::cout << "Read packet " << packet_count << ", size: " << packet->size << std::endl;

            if(filtered_packet->data) {
                av_packet_unref(filtered_packet);
            }
            ret = av_bsf_send_packet(bsf_context, packet);
            if(ret < 0) {
                std::cerr << "Could not send packet to BSF, because: " << AVERROR(ENOMEM) << std::endl;
                break;
            }

            ret = av_bsf_receive_packet(bsf_context, filtered_packet);
            if(ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                std::cerr << "No more filtered packets available right now." << std::endl;
                break;
            } else if(ret < 0) {
                std::cerr << "Could not receive packet from BSF, because: " << AVERROR(ENOMEM) << std::endl;
                break;
            }

            filtered_packet_count++;
            std::cout << "Filtered packet " << filtered_packet_count << ", size: " << filtered_packet->size << std::endl;
            std::cout << "Filtered packet data (first 8 bytes): ";
            for(int i = 0; i < 8 && i < filtered_packet->size; ++i) {
                std::cout << static_cast<int>(filtered_packet->data[i]) << " ";
            }
            std::cout << std::endl;

            filtered_packet->stream_index = output_stream->index;
            av_packet_rescale_ts(filtered_packet, stream->time_base, output_stream->time_base);
            ret = av_interleaved_write_frame(output_format_context, filtered_packet);
            if(ret < 0) {
                std::cerr << "Could not write filtered packet to output file, because: " << AVERROR(ENOMEM) << std::endl;
                break;
            } else {
                std::cout << "Filtered packet written to output file successfully." << std::endl;
            }
        }
    }
 private:
    // 输入文件路径
    std::string input_mp4_ = "";
    AVFormatContext *format_context = nullptr;
    AVStream *stream = nullptr;
    AVCodecContext *codec_context = nullptr;
    AVCodec *codec = nullptr;
    AVPacket* packet = nullptr;
    AVPacket* filtered_packet = nullptr;
    AVBSFContext* bsf_context = nullptr;

    // 输出文件路径
    std::string output_h264_mp4_ = "";
    AVFormatContext *output_format_context = nullptr;
    AVStream *output_stream = nullptr;
    AVCodecContext *output_codec_context = nullptr;
    AVCodec *output_codec = nullptr;

    bool is_inited_ = false;
    std::thread decode_thread_;
};

int main(int argc, char* argv[]) {
    std::string mp4_file = "/mnt/workspace/cgz_workspace/Exercise/camera_example/input/video.mp4";
    std::string output_h264_mp4 = "/mnt/workspace/cgz_workspace/Exercise/camera_example/output/output_h264.h264";

    BSFExample bsf_example(mp4_file, output_h264_mp4);
    if(!bsf_example.init()) {
        std::cerr << "BSFExample init failed!" << std::endl;
        return -1;
    }

    while(true) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    return 0;
}
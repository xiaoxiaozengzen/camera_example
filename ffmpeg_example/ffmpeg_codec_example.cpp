#include <iostream>
#include <string>

extern "C" {
#include "libavformat/avformat.h"
}

/**
 * @brief This function is used to open a video file and print its metadata.
 */
int open_fun() {
    std::string mp4_file = "/mnt/workspace/cgz_workspace/Exercise/camera_example/image/video.mp4";

    AVFormatContext *format_context = nullptr;
    format_context = avformat_alloc_context();
    if(!format_context) {
        std::cerr << "Could not allocate format context, because: " << AVERROR(ENOMEM) << std::endl;
        return -1;
    }

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

    std::cout << "url: " << url << std::endl;
    std::cout << "AV_NOPTS_VALUE: " << AV_NOPTS_VALUE << std::endl;  // 表示无法正确获取数据
    std::cout << "start_time: " << start_time << std::endl;
    std::cout << "AV_TIME_BASE: " << AV_TIME_BASE << std::endl;
    std::cout << "duration: " << duration / double(AV_TIME_BASE) << std::endl;
    std::cout << "nb_streams: " << nb_streams << std::endl;
    for(std::size_t i = 0; i < nb_streams; ++i) {
        AVStream *stream = format_context->streams[i];
        if(stream) {
            std::cout << "Stream " << i << ": codec_type: " << av_get_media_type_string(stream->codecpar->codec_type)
                      << ", codec_id: " << avcodec_get_name(stream->codecpar->codec_id) << std::endl;
        }
    }
    std::cout << "bit_rate: " << bit_rate / 1000 << " kbps" << std::endl;
    std::cout << "iformat_name: " << iformat_name << std::endl;
    std::cout << "iformat_long_name: " << iformat_long_name << std::endl;

    // 关闭输入文件
    avformat_close_input(&format_context);
    // 释放格式上下文
    avformat_free_context(format_context);

    return 0;
}

int main() {
    std::cout << "============================  open_fun ====================== " << std::endl;
    open_fun();

    return 0;
}
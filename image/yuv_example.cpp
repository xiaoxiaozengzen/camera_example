#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "libyuv.h"

#include <opencv2/opencv.hpp>

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
 *     这种存储方式水平和垂直方向上U和V分量的采样率都是Y分量的一半，因此每个像素点的色度信息是由周围的像素点的色度信息计算得到的。
 * 4.实际生产中，格式跟采样格式的对应关系如下：
 *   - YUV420P: YUV的planner格式，采样格式为YUV420，p通常表示planar
 *     - I420: 又叫YU12，即先存储Y，然后存储U，最后存储V的YUV420P格式
 *     - YV12: 即先存储Y，然后存储V，最后存储U的YUV420P格式
 *   - NV12: YUV的semi-planar格式，采样格式为YUV420，即NV开头的一般是semi-planar格式
 *     - NV12：先存储Y分量，然后交替存储UV分量，U在前
 *     - NV21：先存储Y分量，然后交替存储VU分量，V在前
 *   - YUYV422: YUV的packed格式，采样格式为YUV422，YUYV表示存储顺序。不以P结尾的一般是packed格式
 * 5.yuv跟rgb的转换(按照BT.601标准)：
 *   - YUV转RGB公式：
 *     R = Y + 1.402 * (V - 128)
 *     G = Y - 0.344136 * (U - 128) - 0.714136 * (V - 128)
 *     B = Y + 1.772 * (U - 128)
 *     采样：对于YUV420格式：
 *       - Y可以根据每一个像素点直接获取；
 *       - 一个U对应一个block的2x2像素点，一般需要计算四个像素的U，然后再取平均值作为该block中每个像素点的U值；V同理。
 *       - 例如Block(0,0)对应的像素点为(0,0),(0,1),(1,0),(1,1)
 *   - RGB转YUV公式：
 *     Y = 0.299 * R + 0.587 * G + 0.114 * B
 *     U = -0.14713 * R - 0.28886 * G + 0.436 * B + 128
 *     V = 0.615 * R - 0.51499 * G - 0.10001 * B + 128
 *   注意：在计算UV分量中，+128是因为UV会有负值，加上128后可以将其值域映射到0-255之间，方便存储和处理。
 *        所以在YUV转RGB时，需要将U和V分量减去128以还原其原始值。
 */

void yuv_info(std::string yuv_file, int width, int height, int frame_size) {
    std::cout << "width: " << width << ", height: " << height << ", frame_size: " << frame_size << std::endl;

    std::ifstream yuv_stream(yuv_file, std::ios::binary);
    if(!yuv_stream.is_open()) {
        std::cerr << "Could not open YUV file: " << yuv_file << std::endl;
        return;
    }
    std::stringstream yuv_ss;
    yuv_ss << yuv_stream.rdbuf();

    std::string yuv_data = yuv_ss.str();
    int yuv_data_size = yuv_data.size();
    if(yuv_data_size != frame_size) {
        std::cerr << "Expected frame size: " << frame_size << " bytes, but got: " << yuv_data_size << " bytes." << std::endl;
        return;
    } else {
        std::cout << "YUV data size matches expected frame size: " << yuv_data_size << " bytes." << std::endl;
    }

    std::vector<unsigned char> y_data(yuv_data.begin(), yuv_data.begin() + width * height);
    std::cout << "Y data size: " << y_data.size() << " bytes." << std::endl;
    std::vector<unsigned char> u_data(yuv_data.begin() + width * height, yuv_data.begin() + width * height + (width / 2) * (height / 2));
    std::cout << "U data size: " << u_data.size() << " bytes." << std::endl;
    std::vector<unsigned char> v_data(yuv_data.begin() + width * height + (width / 2) * (height / 2), yuv_data.end());
    std::cout << "V data size: " << v_data.size() << " bytes." << std::endl;
}

/**
 * @brief Convert YUV420P format to grayscale YUV format.
 * @note Y值正常写入，U和V分量填充为128。
 */
void yuv2gray(std::string yuv_file_in, std::string yuv_file_out, int width, int height, int frame_size) {
    std::cout << "width: " << width << ", height: " << height << ", frame_size: " << frame_size << std::endl;

    // Open the YUV file
    std::ifstream yuv_stream(yuv_file_in, std::ios::binary);
    if(!yuv_stream.is_open()) {
        std::cerr << "Could not open YUV file: " << yuv_file_in << std::endl;
        return;
    }

    // Check if the file is a YUV file by looking for the ".yuv" suffix
    std::string subfix = ".yuv";
    std::string::size_type pos = yuv_file_in.find(subfix);
    if(pos == std::string::npos) {
        std::cerr << "The file is not a YUV file: " << yuv_file_in << std::endl;
        return;
    }
    
    std::string gray_yuv_file = yuv_file_out;

    // Open the gray YUV file
    std::ofstream gray_yuv_stream(gray_yuv_file, std::ios::binary);
    if(!gray_yuv_stream.is_open()) {
        std::cerr << "Could not open output YUV file: " << gray_yuv_file << std::endl;
        return;
    }

    std::stringstream yuv_ss;
    yuv_ss << yuv_stream.rdbuf();

    std::string yuv_data = yuv_ss.str();
    int yuv_data_size = yuv_data.size();
    if(yuv_data_size != frame_size) {
        std::cerr << "Expected frame size: " << frame_size << " bytes, but got: " << yuv_data_size << " bytes." << std::endl;
        return;
    } else {
        std::cout << "YUV data size matches expected frame size: " << yuv_data_size << " bytes." << std::endl;
    }

    std::vector<unsigned char> y_data(yuv_data.begin(), yuv_data.begin() + width * height);
    std::cout << "Y data size: " << y_data.size() << " bytes." << std::endl;
    std::vector<unsigned char> u_data(yuv_data.begin() + width * height, yuv_data.begin() + width * height + (width / 2) * (height / 2));
    std::cout << "U data size: " << u_data.size() << " bytes." << std::endl;
    std::vector<unsigned char> v_data(yuv_data.begin() + width * height + (width / 2) * (height / 2), yuv_data.end());
    std::cout << "V data size: " << v_data.size() << " bytes." << std::endl;

    // Write the Y data to the gray YUV file
    gray_yuv_stream.write(reinterpret_cast<const char*>(y_data.data()), y_data.size());
    std::vector<unsigned char> zero_data(width * height / 4, 128); // U和V分量填充为128
    gray_yuv_stream.write(reinterpret_cast<const char*>(zero_data.data()), zero_data.size());
    gray_yuv_stream.write(reinterpret_cast<const char*>(zero_data.data()), zero_data.size());
}

void yuv2jpg(std::string yuv_file, std::string jpeg_file, int width, int height, int frame_size) {
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
    std::string jpg_file = jpeg_file;
    std::cout << "------ jpg file: " << jpg_file << std::endl;
    
    std::vector<unsigned char> yuv_data(frame_size);
    yuv_stream.read(reinterpret_cast<char*>(yuv_data.data()), frame_size);
    if(yuv_stream.gcount() != frame_size) {
        std::cerr << "Error reading YUV file: " << yuv_file << std::endl;
        return;
    }

    /**
     * 1. 前h行，w宽，是存储Y分量数据，共h*w字节
     * 2. 后h/2行，w宽，交替存放U和V分量数据，共h*w/2字节
     *   - 在I420格式中，U和V是分开存储的，先存U平面，然后再存V平面
     *   - 在I420格式中，U和V。各占h/2行，w/2宽
     *   - 把他们排列成W宽的单通道图像数据时，需要把每行的宽度扩展为W，则U和V分别占h/4行，w宽
     */
    cv::Mat yuv_image(height + height / 2, width, CV_8UC1, yuv_data.data());
    cv::Mat rgb_image;
    cv::cvtColor(yuv_image, rgb_image, cv::COLOR_YUV2BGR_I420); // YUV420格式转换为BGR格式

    cv::imwrite(jpg_file, rgb_image);
}

void yuv2rgb(std::string yuv_file, std::string rgb_file, std::string bgr_file, int width, int height, int frame_size) {
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

    std::cout << "------ rgb file: " << rgb_file << std::endl;
    std::cout << "------ bgr file: " << bgr_file << std::endl;

    // Read YUV data
    std::vector<unsigned char> yuv_data(frame_size);
    yuv_stream.read(reinterpret_cast<char*>(yuv_data.data()), frame_size);
    if(yuv_stream.gcount() != frame_size) {
        std::cerr << "Error reading YUV file: " << yuv_file << std::endl;
        return;
    }

    // yuv info
    uint8_t* yptr = yuv_data.data();
    uint8_t* uptr = yuv_data.data() + width * height;
    uint8_t* vptr = uptr + (width / 2) * (height / 2);
    int y_stride = width;
    int u_stride = width / 2;
    int v_stride = width / 2;

    // yuv to rgb using libyuv
    std::vector<unsigned char> rgb_data(width * height * 3);
    std::vector<unsigned char> bgr_data(width * height * 3);
    int rgb_stride = width * 3;
    // 注意：该函数中，RGB24在内存中的存储是BGR顺序
    int ret = libyuv::I420ToRGB24(
        yptr, y_stride,
        uptr, u_stride,
        vptr, v_stride,
        rgb_data.data(), rgb_stride,
        width, height
    );
    if(ret != 0) {
        std::cerr << "Error converting YUV to RGB using libyuv." << std::endl;
        return;
    }

    // 注意：该函数中，是按照RGB顺序存储的
    ret = libyuv::I420ToRAW(
        yptr, y_stride,
        uptr, u_stride,
        vptr, v_stride,
        bgr_data.data(), rgb_stride,
        width, height
    );
    if(ret != 0) {
        std::cerr << "Error converting YUV to BGR using libyuv." << std::endl;
        return;
    }

    // print rgb data size
    cv::Mat rgb_image(height, width, CV_8UC3, rgb_data.data());
    cv::imwrite(rgb_file, rgb_image);

    cv::Mat bgr_image(height, width, CV_8UC3, bgr_data.data());
    cv::Mat bgr_image_converted;
    cv::cvtColor(bgr_image, bgr_image_converted, cv::COLOR_RGB2BGR);
    cv::imwrite(bgr_file, bgr_image_converted);
}

void yuv2rgb_math(std::string yuv_file, std::string bgr_file, int width, int height, int frame_size) {
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

    std::cout << "------ bgr file: " << bgr_file << std::endl;

    // Read YUV data
    std::vector<unsigned char> yuv_data(frame_size);
    yuv_stream.read(reinterpret_cast<char*>(yuv_data.data()), frame_size);
    if(yuv_stream.gcount() != frame_size) {
        std::cerr << "Error reading YUV file: " << yuv_file << std::endl;
        return;
    }

    // yuv info
    uint8_t* yptr = yuv_data.data();
    uint8_t* uptr = yuv_data.data() + width * height;
    uint8_t* vptr = uptr + (width / 2) * (height / 2);
    int y_stride = width;
    int u_stride = width / 2;
    int v_stride = width / 2;

    // bgr info
    std::vector<unsigned char> bgr_data(width * height * 3);
    int bgr_stride = width * 3;

    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            int bgr_index = i * bgr_stride + j * 3;

            unsigned char Y = *(yptr + i * y_stride + j);
            unsigned char U = *(uptr + (i / 2) * u_stride + (j / 2));
            unsigned char V = *(vptr + (i / 2) * v_stride + (j / 2));

            float R = Y + 1.402 * (V - 128);
            float G = Y - 0.344136 * (U - 128) - 0.714136 * (V - 128);
            float B = Y + 1.772 * (U - 128);

            // Clamp values to [0, 255]
            R = std::min(std::max(R, 0.0f), 255.0f);
            G = std::min(std::max(G, 0.0f), 255.0f);
            B = std::min(std::max(B, 0.0f), 255.0f);

            bgr_data[bgr_index + 0] = static_cast<unsigned char>(B);
            bgr_data[bgr_index + 1] = static_cast<unsigned char>(G);
            bgr_data[bgr_index + 2] = static_cast<unsigned char>(R);
        }
    }

    cv::Mat bgr_image(height, width, CV_8UC3, bgr_data.data());
    cv::imwrite(bgr_file, bgr_image);
}

int main() {
    std::cout << "============================  yuv_info 1 ====================== " << std::endl;
    std::string yuv_file = "/mnt/workspace/cgz_workspace/Exercise/camera_example/input/650_yuv420p.yuv";
    int width = 1080;
    int height = 1920;
    int frame_size = width * height * 3 / 2; // YUV420P格式
    yuv_info(yuv_file, width, height, frame_size);

    std::cout << "=============================  yuv_info 2 ====================== " << std::endl;
    std::string yuv_file2 = "/mnt/workspace/cgz_workspace/Exercise/camera_example/input/video_single_frame.yuv";
    int width2 = 960;
    int height2 = 540;
    int frame_size2 = width2 * height2 * 3 / 2; // YUV420P格式
    yuv_info(yuv_file2, width2, height2, frame_size2);

    std::cout << "============================  yuv2gray ====================== " << std::endl;
    std::string yuv_file3 = "/mnt/workspace/cgz_workspace/Exercise/camera_example/input/video_single_frame.yuv";
    std::string yuv_file3_out = "/mnt/workspace/cgz_workspace/Exercise/camera_example/output/video_single_frame_gray.yuv";
    int width3 = 960;
    int height3 = 540;
    int frame_size3 = width3 * height3 * 3 / 2; // YUV420P格式
    yuv2gray(yuv_file3, yuv_file3_out, width3, height3, frame_size3);

    std::cout << "============================  yuv2jpg ====================== " << std::endl;
    std::string yuv_file4 = "/mnt/workspace/cgz_workspace/Exercise/camera_example/input/650_yuv420p.yuv";
    std::string jpeg_file4 = "/mnt/workspace/cgz_workspace/Exercise/camera_example/output/650_yuv420p.jpg";
    yuv2jpg(yuv_file4, jpeg_file4, 1080, 1920, 1080 * 1920 * 3 / 2); // YUV420P格式
    std::string yuv_file5 = "/mnt/workspace/cgz_workspace/Exercise/camera_example/input/video_single_frame.yuv";
    std::string jpeg_file5 = "/mnt/workspace/cgz_workspace/Exercise/camera_example/output/video_single_frame.jpg";
    yuv2jpg(yuv_file5, jpeg_file5, 960, 540, 960 * 540 * 3 / 2); // YUV420P格式
    std::string yuv_gray_file = "/mnt/workspace/cgz_workspace/Exercise/camera_example/input/video_single_frame_gray.yuv";
    std::string jpeg_gray_file = "/mnt/workspace/cgz_workspace/Exercise/camera_example/output/video_single_frame_gray.jpg";
    yuv2jpg(yuv_gray_file, jpeg_gray_file, 960, 540, 960 * 540 * 3 /2); // YUV灰度格式

    std::cout << "============================  yuv2rgb ====================== " << std::endl;
    std::string yuv_file6 = "/mnt/workspace/cgz_workspace/Exercise/camera_example/input/650_yuv420p.yuv";
    std::string rgb_file6 = "/mnt/workspace/cgz_workspace/Exercise/camera_example/output/650_yuv420p_rgb_libyuv.jpg";
    std::string bgr_file6 = "/mnt/workspace/cgz_workspace/Exercise/camera_example/output/650_yuv420p_bgr_libyuv.jpg";
    yuv2rgb(yuv_file6, rgb_file6, bgr_file6, 1080, 1920, 1080 * 1920 * 3 / 2); // YUV420P格式

    std::cout << "============================  yuv2rgb_math ====================== " << std::endl;
    std::string yuv_file7 = "/data/workspace/Exercise/camera_example/input/650_yuv420p.yuv";
    std::string bgr_file7 = "/data/workspace/Exercise/camera_example/output/650_yuv420p_bgr_math.jpg";
    yuv2rgb_math(yuv_file7, bgr_file7, 1080, 1920, 1080 * 1920 * 3 / 2); // YUV420P格式

    return 0;
}
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

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
void yuv2gray(std::string yuv_file, int width, int height, int frame_size) {
    std::cout << "width: " << width << ", height: " << height << ", frame_size: " << frame_size << std::endl;

    // Open the YUV file
    std::ifstream yuv_stream(yuv_file, std::ios::binary);
    if(!yuv_stream.is_open()) {
        std::cerr << "Could not open YUV file: " << yuv_file << std::endl;
        return;
    }

    // Check if the file is a YUV file by looking for the ".yuv" suffix
    std::string subfix = ".yuv";
    std::string::size_type pos = yuv_file.find(subfix);
    if(pos == std::string::npos) {
        std::cerr << "The file is not a YUV file: " << yuv_file << std::endl;
        return;
    }
    std::string file_without_subfix = yuv_file.substr(0, pos);
    std::cout << "file_without_subfix: " << file_without_subfix << std::endl;
    std::string gray_yuv_file = file_without_subfix + "_gray.yuv";

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
 */
void yuv2jpg(std::string yuv_file, int width, int height, int frame_size) {
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
    std::string yuv_file3 = "/mnt/workspace/cgz_workspace/Exercise/camera_example/output/video_single_frame.yuv";
    int width3 = 960;
    int height3 = 540;
    int frame_size3 = width3 * height3 * 3 / 2; // YUV420P格式
    yuv2gray(yuv_file3, width3, height3, frame_size3);

    std::cout << "============================  yuv2jpg ====================== " << std::endl;
    std::string yuv_file4 = "/mnt/workspace/cgz_workspace/Exercise/camera_example/output/650_yuv420p.yuv";
    yuv2jpg(yuv_file4, 1080, 1920, 1080 * 1920 * 3 / 2); // YUV420P格式
    std::string yuv_file5 = "/mnt/workspace/cgz_workspace/Exercise/camera_example/output/video_single_frame.yuv";
    yuv2jpg(yuv_file5, 960, 540, 960 * 540 * 3 / 2); // YUV420P格式
    std::string yuv_gray_file = "/mnt/workspace/cgz_workspace/Exercise/camera_example/output/video_single_frame_gray.yuv";
    yuv2jpg(yuv_gray_file, 960, 540, 960 * 540 * 3 /2); // YUV灰度格式
    std::string yuv_file6 = "/mnt/workspace/cgz_workspace/Exercise/camera_example/output/frame_1.yuv";
    yuv2jpg(yuv_file6, 960, 540, 960 * 540 * 3 /2); // YUV灰度格式

    return 0;
}
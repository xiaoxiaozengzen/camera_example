#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

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


    return 0;
}
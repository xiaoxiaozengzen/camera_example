#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>
#include <cuda.h>

int main(int argc, char** argv) {
    std::string image_input_yuv = "/mnt/workspace/cgz_workspace/Exercise/camera_example/input/650_yuv420p.yuv";
    int width = 1080;
    int height = 1920;
    size_t img_bytes = width * height * 3 / 2;

    std::ifstream infile(image_input_yuv, std::ios::binary | std::ios::ate);
    if(!infile.is_open()) {
        std::cerr << "Failed to open input file: " << image_input_yuv << std::endl;
        return -1;
    }

    std::size_t data_size = infile.tellg();
    std::cout << "data_size: " << data_size
              << ", width * height * 3 / 2: " << img_bytes
              << std::endl;
    infile.seekg(0, std::ios::beg);
    std::vector<char> buffer(data_size);
    infile.read(buffer.data(), data_size);
    infile.close();



    std::cout << "Success!" << std::endl;
}
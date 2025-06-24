#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <cstring>

int main() {
    std::string root_path = "/mnt/workspace/cgz_workspace/Exercise/camera_example";
    std::string file_name = root_path + "/image/650.jpeg";

    std::ifstream file(file_name, std::ios::binary);
    if(!file.is_open()) {
        std::cerr << "Error: Could not open file " << file_name << std::endl;
        return 1;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    file.close();

    std::string file_content = buffer.str();

    std::cout << "buffer size: " << buffer.str().size() << " bytes"
              << ", " << buffer.str().size() / 1024.0 << " KB"
              << ", " << buffer.str().size() / (1024.0 * 1024.0) << " MB"
              << std::endl;

    std::vector<unsigned char> data(file_content.begin(), file_content.end());

    // 00~01
    std::cout << "0 bytes: " << std::hex << static_cast<std::uint32_t>(data[0]) << std::dec << std::endl;
    std::cout << "1 bytes: " << std::hex << static_cast<std::uint32_t>(data[1]) << std::dec << std::endl;
    std::uint16_t first_marker;
    std::uint16_t first_marker_low = data[1] << 0 & 0xFF;
    std::uint16_t first_marker_high = data[0] << 8 & 0xFF00;
    first_marker = first_marker_low | first_marker_high;
    std::cout << "first marker: " << std::hex << first_marker << std::dec << std::endl;

    // 02~03 04~05
    std::cout << "2 bytes: " << std::hex << static_cast<std::uint32_t>(data[2]) << std::dec << std::endl;
    std::cout << "3 bytes: " << std::hex << static_cast<std::uint32_t>(data[3]) << std::dec << std::endl;
    std::uint16_t second_marker = data[2] << 8 & 0xFF00 | data[3] << 0 & 0x00FF;
    std::cout << "second marker: " << std::hex << second_marker << std::dec << std::endl;
    std::cout << "4 bytes: " << std::hex << static_cast<std::uint32_t>(data[4]) << std::dec << std::endl;
    std::cout << "5 bytes: " << std::hex << static_cast<std::uint32_t>(data[5]) << std::dec << std::endl;
    std::uint16_t second_marker_length = data[4] << 8 & 0xFF00 | data[5] << 0 & 0x00FF;
    std::cout << "second marker length: " << second_marker_length << std::endl;

}
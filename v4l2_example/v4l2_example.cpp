extern "C" {
    #include <linux/videodev2.h>
}
#include <fcntl.h>
#include <iostream>
#include <sys/ioctl.h>
#include <unistd.h>
#include <cstring>
#include <sys/mman.h>

/**
 * 可以使用ioctl命令查询视频设备的能力。具体可以参考文件linux/videodev2.h中定义的：
 * #define VIDIOC_QUERYCAP		 _IOR('V',  0, struct v4l2_capability)
 * #define VIDIOC_ENUM_FMT         _IOWR('V',  2, struct v4l2_fmtdesc)
 * ......
 * 
 * 部分请求的定义如下：
 * VIDIOC_QUERYCAP: 查询视频设备的能力。
 * VIDIOC_ENUM_FMT: 查询当前视频设备支持的格式。
 * VIDIOC_G_FMT: 获取当前视频设备的格式。
 */

void VIDIOC_QUERYCAP_example(int fd, struct v4l2_capability *cap) {
    /**
     * struct v4l2_capability {
     *     __u8    driver[16];      // 驱动名称
     *     __u8    card[32];        // 设备名称（如摄像头型号）
     *     __u8    bus_info[32];    // 总线信息（如 USB 端口、PCI 位置等）
     *     __u32   version;         // 驱动版本号
     *     __u32   capabilities;    // 设备支持的功能（位掩码）
     *     __u32   device_caps;     // 设备自身的功能（位掩码）
     *     __u32   reserved[3];     // 保留，未用
     * };
     */
    int ret = ioctl(fd, VIDIOC_QUERYCAP, cap);
    if( ret < 0) {
        perror("VIDIOC_QUERYCAP");
        return;
    }
    std::cout << "Driver: " << cap->driver << std::endl;
    std::cout << "Card: " << cap->card << std::endl;
    std::cout << "Bus Info: " << cap->bus_info << std::endl;
    std::cout << "Version: " << cap->version << std::endl;
    /**
     * 设备支持的功能集合：
     *   - V4L2_CAP_VIDEO_CAPTURE: 支持视频采集
     *   - V4L2_CAP_VIDEO_OUTPUT: 支持视频输出
     *   - V4L2_CAP_VIDEO_STREAMING: 支持视频流处理
     */
    std::cout << "Capabilities: " << std::hex << cap->capabilities << std::dec << std::endl;
    /**
     * 设备自身的功能集合，用于区分设备本身能力跟驱动能力的区别。
     *   - V4L2_CAP_DEVICE_CAPS: 设备能力
     *   - V4L2_CAP_STREAMING: 支持流式处理
     *   - V4L2_CAP_READWRITE: 支持读写操作
     *   - V4L2_CAP_ASYNCIO: 支持异步I/O
     */
    std::cout << "Device Caps: " << std::hex << cap->device_caps << std::dec << std::endl;
}

/**
 * struct v4l2_fmtdesc {
 *      __u32   index;          // format索引，从0开始
 *      __u32   type;           // 设备类型，enum v4l2_buf_type
 *      __u32   flags;          // 格式标志
 *      __u32   pixelformat;   // format fourcc编码(Four Character Code)
 *      __u8    description[32]; // 格式描述
 *      __u32   reserved[4];    // 保留字段
 * };
 */
void VIDIOC_ENUM_FMT_example(int fd) {
    struct v4l2_fmtdesc fmt;
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

    int index = 0;

    while(true) {
        fmt.index = index;
        index++;
        int ret = ioctl(fd, VIDIOC_ENUM_FMT, &fmt);
        if (ret < 0) {
            break;
        }

        std::cout << "index: " << fmt.index << std::endl;
        const char *fourcc = reinterpret_cast<const char*>(&fmt.pixelformat);
        std::cout << "pixelformat: " 
                  << fourcc[0] << fourcc[1] << fourcc[2] << fourcc[3] << std::endl;
        std::cout << "Description: " << fmt.description << std::endl;

        /**
         * Flags:
         *  - 0: 无特殊标志
         *  - V4L2_FMT_FLAG_COMPRESSED: 压缩格式
         *  - V4L2_FMT_FLAG_EMULATED: 模拟格式
         */
        std::cout << "Flags: " << fmt.flags << std::endl;
    }
}

/**
 * struct v4l2_format {
 *     __u32   type;           // 设备类型，enum v4l2_buf_type
 *     union {
 *         struct v4l2_pix_format   pix;    // 像素格式
 *         struct v4l2_window       win;    // 窗口格式
 *         struct v4l2_vbi_format   vbi;    // VBI格式
 *         struct v4l2_sliced_vbi_format sliced; // 切片VBI格式
 *         struct v4l2_meta_format   meta;   // 元数据格式
 *         struct v4l2_sdr_format    sdr;    // SDR格式
 *         struct v4l2_dv_format     dv;     // 数字视频格式
 *     } fmt;                     // 格式联合体
 * };
 */
void VIDIOC_S_FMT_example(int fd) {
    struct v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt)); // 清空格式结构体

    struct v4l2_pix_format pix_fmt;
    pix_fmt.width = 640;          // 设置宽度
    pix_fmt.height = 480;         // 设置高度
    pix_fmt.pixelformat = V4L2_PIX_FMT_YUYV; // 设置像素格式
    // pix_fmt.field = V4L2_FIELD_NONE; // 设置场类型
    // pix_fmt.bytesperline = 0;     // 字节行数，0表示自动计算
    // pix_fmt.sizeimage = 0;        // 图像大小，0表示自动计算
    // pix_fmt.colorspace = V4L2_COLORSPACE_SRGB; // 设置颜色空间
    // pix_fmt.priv = 0;             // 私有数据，通常为0
    // pix_fmt.flags = 0;            // 格式标志，通常为0
    // pix_fmt.ycbcr_enc = V4L2_YCBCR_ENC_DEFAULT; // YCbCr编码方式
    // pix_fmt.quantization = V4L2_QUANTIZATION_DEFAULT; // 量化方式
    // pix_fmt.xfer_func = V4L2_XFER_FUNC_DEFAULT; // 传输函数

    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE; // 设置设备类型
    fmt.fmt.pix = pix_fmt; // 将像素格式设置到格式联合体

    int ret = ioctl(fd, VIDIOC_S_FMT, &fmt);
    if (ret < 0) {
        perror("VIDIOC_S_FMT");
        return;
    }
    
    memset(&fmt, 0, sizeof(struct v4l2_format));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE; // 设置设备类型
    ret = ioctl(fd, VIDIOC_G_FMT, &fmt);
    if (ret < 0) {
        perror("VIDIOC_G_FMT");
        return;
    }
    std::cout << "Width: " << fmt.fmt.pix.width << std::endl;
    std::cout << "Height: " << fmt.fmt.pix.height << std::endl;
    const char *fourcc = reinterpret_cast<const char*>(&fmt.fmt.pix.pixelformat);
    std::cout << "Pixel Format: "
              << fourcc[0] << fourcc[1] << fourcc[2] << fourcc[3] << std::endl;
    std::cout << "Field: "
              << (fmt.fmt.pix.field == V4L2_FIELD_NONE ? "None"
                  : fmt.fmt.pix.field == V4L2_FIELD_TOP ? "Top"
                  : fmt.fmt.pix.field == V4L2_FIELD_BOTTOM ? "Bottom"
                  : fmt.fmt.pix.field == V4L2_FIELD_INTERLACED ? "Interlaced"
                  : "Unknown") << std::endl;
    std::cout << "Bytes per Line: " << fmt.fmt.pix.bytesperline << std::endl;
    std::cout << "Size Image: " << fmt.fmt.pix.sizeimage << std::endl;
    std::cout << "Colorspace: "
              << (fmt.fmt.pix.colorspace == V4L2_COLORSPACE_SRGB ? "sRGB"
                  : fmt.fmt.pix.colorspace == V4L2_COLORSPACE_JPEG ? "JPEG"
                  : fmt.fmt.pix.colorspace == V4L2_COLORSPACE_SMPTE170M ? "SMPTE170M"
                  : fmt.fmt.pix.colorspace == V4L2_COLORSPACE_SMPTE240M ? "SMPTE240M"
                  : "Unknown") << std::endl;
    std::cout << "YCbCr Encoding: "
              << (fmt.fmt.pix.ycbcr_enc == V4L2_YCBCR_ENC_DEFAULT ? "Default"
                  : fmt.fmt.pix.ycbcr_enc == V4L2_YCBCR_ENC_601 ? "601"
                  : fmt.fmt.pix.ycbcr_enc == V4L2_YCBCR_ENC_709 ? "709"
                  : fmt.fmt.pix.ycbcr_enc == V4L2_YCBCR_ENC_BT2020 ? "BT2020"
                  : fmt.fmt.pix.ycbcr_enc == V4L2_YCBCR_ENC_SMPTE240M ? "SMPTE240M"
                  : "Unknown") << std::endl;
    std::cout << "Quantization: "
              << (fmt.fmt.pix.quantization == V4L2_QUANTIZATION_DEFAULT ? "Default"
                  : fmt.fmt.pix.quantization == V4L2_QUANTIZATION_LIM_RANGE ? "Limited Range"
                  : "Unknown") << std::endl;
    std::cout << "Transfer Function: "
              << (fmt.fmt.pix.xfer_func == V4L2_XFER_FUNC_DEFAULT ? "Default"
                  : fmt.fmt.pix.xfer_func == V4L2_XFER_FUNC_709 ? "709"
                  : fmt.fmt.pix.xfer_func == V4L2_XFER_FUNC_SRGB ? "sRGB"
                  : fmt.fmt.pix.xfer_func == V4L2_XFER_FUNC_SMPTE240M ? "SMPTE240M"
                  : "Unknown") << std::endl;
}

void capture_picture_example(int fd) {
    /**
     * struct v4l2_requestbuffers {
     *     __u32   count;          // 请求的缓冲区数量
     *     __u32   type;           // 缓冲区类型，enum v4l2_buf_type
     *     __u32   memory;         // 内存映射方式，enum v4l2_memory
     *     __u32   reserved[2];    // 保留字段
     * };
     */
    struct v4l2_requestbuffers req;
    memset(&req, 0, sizeof(req));
    req.count = 4; // 请求4个缓冲区
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE; // 设置缓冲区类型
    req.memory = V4L2_MEMORY_MMAP; // 设置内存映射方式
    int ret = ioctl(fd, VIDIOC_REQBUFS, &req);
    if (ret < 0) {
        perror("VIDIOC_REQBUFS");
        return;
    }

    /**
     * struct v4l2_buffer {
     *     __u32   index;          // 缓冲区索引
     *     __u32   type;           // 缓冲区类型，enum v4l2_buf_type
     *     __u32   bytesused;      // 已使用的字节数
     *     __u32   flags;          // 缓冲区标志
     *     __u32   field;          // 场类型，enum v4l2_field
     *     __u32   timestamp;      // 时间戳
     *     struct v4l2_timecode timecode; // 时间码
     *     __u32   sequence;       // 序列号
     *     __u32   memory;         // 内存映射方式，enum v4l2_memory
     *     union {
     *         __u32   offset;    // 内存映射偏移量
     *         void    *userptr; // 用户指针
     *         struct v4l2_plane planes[VIDEO_MAX_PLANES]; // 平面格式
     *     } m;                    // 内存映射联合体
     *     __u32   length;        // 缓冲区长度
     *     __u32   reserved2;     // 保留字段
     *     __u32   reserved[2];   // 保留字段
     * };
     */
    struct v4l2_buffer mmap_buf;
    mmap_buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE; // 设置缓冲区类型
    char* mmpader[4]; // 用于存储内存映射的指针
    int buf_size[4]; // 用于存储每个缓冲区的大小
    for(int i = 0; i < 4; i++) {
        mmap_buf.index = i;
        ret = ioctl(fd, VIDIOC_QUERYBUF, &mmap_buf);
        if(ret < 0) {
            perror("VIDIOC_QUERYBUF");
        }
        std::cout << "Buffer " << i << ": " << std::endl;
        mmpader[i] = (char*)mmap(NULL, mmap_buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, mmap_buf.m.offset);
        buf_size[i] = mmap_buf.length;
        ret = ioctl(fd, VIDIOC_QBUF, &mmap_buf);
        if(ret < 0) {
            perror("VIDIOC_QBUF");
        }
    }

    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ret = ioctl(fd, VIDIOC_STREAMON, &type);
    if (ret < 0) {
        perror("VIDIOC_STREAMON");
        return;
    }

    struct v4l2_buffer readbuffer;
    readbuffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ret = ioctl(fd, VIDIOC_DQBUF, &readbuffer);
    if (ret < 0) {
        perror("VIDIOC_DQBUF");
        return;
    }

    FILE* fp = fopen("output.yuv", "w+");
    if (!fp) {
        perror("Failed to open output file");
        return;
    }
    fwrite(mmpader[readbuffer.index], 1, readbuffer.bytesused, fp);
    fclose(fp);

    ret = ioctl(fd, VIDIOC_QBUF, &readbuffer);
    if (ret < 0) {
        perror("VIDIOC_QBUF");
        return;
    }

    ret = ioctl(fd, VIDIOC_STREAMOFF, &type);
    if (ret < 0) {
        perror("VIDIOC_STREAMOFF");
        return;
    }

    for(int i = 0; i < 4; i++) {
        if (mmpader[i]) {
            munmap(mmpader[i], buf_size[i]);
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <video_device>" << std::endl;
        return 1;
    }

    const char *video_device = argv[1];
    int fd = open(video_device, O_RDWR, 0);
    if (fd < 0) {
        perror("Opening video device");
        return 1;
    }

    std::cout << "===================== VIDIOC_QUERYCAP ====================" << std::endl;
    struct v4l2_capability cap;
    VIDIOC_QUERYCAP_example(fd, &cap);

    std::cout << "===================== VIDIOC_ENUM_FMT ====================" << std::endl;
    VIDIOC_ENUM_FMT_example(fd);

    std::cout << "===================== VIDIOC_S_FMT ====================" << std::endl;
    struct v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    VIDIOC_S_FMT_example(fd);

    std::cout << "===================== capture picture ====================" << std::endl;
    capture_picture_example(fd);

    close(fd);
    return 0;
}
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>

#include <string.h>
#include <dirent.h>  
#include <sys/stat.h>
#include <sys/types.h>

#include <cuda_runtime_api.h>
#include <nvjpeg.h>
#include <nppi_geometry_transforms.h>


#define CHECK_CUDA(call)                                                        \
    {                                                                           \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess)                                                  \
        {                                                                       \
            std::cout << "CUDA Runtime failure: '#" << _e << "' at " <<  __FILE__ << ":" << __LINE__ << std::endl;\
            exit(1);                                                            \
        }                                                                       \
    }

#define CHECK_NVJPEG(call)                                                      \
    {                                                                           \
        nvjpegStatus_t _e = (call);                                             \
        if (_e != NVJPEG_STATUS_SUCCESS)                                        \
        {                                                                       \
            std::cout << "NVJPEG failure: '#" << _e << "' at " <<  __FILE__ << ":" << __LINE__ << std::endl;\
            exit(1);                                                            \
        }                                                                       \
    }

struct image_resize_params_t {
  std::string input_dir;
  std::string output_dir;
  int quality;
  int width;
  int height;
  int dev;
};

/**
 * struct NppiSize {
 *     int width;  // 图像宽度
 *     int height; // 图像高度
 * };
 *
 * struct NppiRect {
 *     int x;      // 矩形左上角的X坐标
 *     int y;      // 矩形左上角的Y坐标
 *     int width;  // 矩形的宽度
 *     int height; // 矩形的高度
 * };
 *
 * struct nvjpegImage_t {
 *     unsigned char* channel[NVJPEG_MAX_COMPONENT]; // 每个通道的数据指针
 *     size_t pitch[NVJPEG_MAX_COMPONENT];           // 每个通道的行跨度（以字节为单位）
 * };
 */
typedef struct {
    NppiSize size;
    nvjpegImage_t data;
} image_t;


int dev_malloc(void** p, size_t s)
{
    return (int)cudaMalloc(p, s);
}

int dev_free(void* p)
{
    return (int)cudaFree(p);
}

/**
 * @brief 判断nvjpegOutputFormat_t的格式
 * @note 这是一个枚举类型，定义在nvjpeg.h中
 *     - NVJPEG_OUTPUT_RGBI：interleaved RGB格式，即按照像素RGB,RGB...顺序存储
 *     - NVJPEG_OUTPUT_RGB: planar RGB格式，即先存储所有R分量，再存储所有G分量，最后存储所有B分量
 */
bool is_interleaved(nvjpegOutputFormat_t format)
{
    if (format == NVJPEG_OUTPUT_RGBI || format == NVJPEG_OUTPUT_BGRI)
        return true;
    else
        return false;
}

std::string subsamplingToString(int subsampling)
{
    switch(subsampling)
    {
        case static_cast<int>(NVJPEG_CSS_444):
            return "NVJPEG_CSS_444";
        case static_cast<int>(NVJPEG_CSS_422):
            return "NVJPEG_CSS_422";
        case static_cast<int>(NVJPEG_CSS_420):
            return "NVJPEG_CSS_420";
        case static_cast<int>(NVJPEG_CSS_440):
            return "NVJPEG_CSS_440";
        case static_cast<int>(NVJPEG_CSS_411):
            return "NVJPEG_CSS_411";
        case static_cast<int>(NVJPEG_CSS_410):
            return "NVJPEG_CSS_410";
        default:
            return "Unknown";
    }
}


// *****************************************************************************
// nvJPEG handles and parameters
// -----------------------------------------------------------------------------
/**
 * enum nvjpegBackend_t {
 *     NVJPEG_BACKEND_DEFAULT = 0,        // 默认后端
 *     NVJPEG_BACKEND_HYBRID = 1,         // 使用CPU进行Huffman解码
 *     NVJPEG_BACKEND_GPU_HYBRID = 2,     // 使用GPU辅助进行Huffman解码， nvjpegDecodeBatched会基于GPU进行Huffman解码
 *     NVJPEG_BACKEND_HARDWARE = 3         // 使用专用硬件加速器进行解码
 *     NVJPEG_BACKEND_GPU_HYBRID_DEVICE = 4 // nvjpegDecodeBatched支持bitstream直接从设备内存读取
 *     NVJPEG_BACKEND_HARDWARE_DEVICE = 5   // nvjpegDecodeBatched支持bitstream直接从设备内存读取
 * };
 */
nvjpegBackend_t impl = NVJPEG_BACKEND_GPU_HYBRID;
nvjpegHandle_t nvjpeg_handle;
nvjpegJpegStream_t nvjpeg_jpeg_stream;
nvjpegDecodeParams_t nvjpeg_decode_params;
nvjpegJpegState_t nvjpeg_decoder_state;
nvjpegEncoderParams_t nvjpeg_encode_params;
nvjpegEncoderState_t nvjpeg_encoder_state;


// *****************************************************************************
// Decode, Resize and Encoder function
// -----------------------------------------------------------------------------
int decodeResizeEncodeOneImage(std::string sImagePath, std::string sOutputPath, double &time, int resizeWidth, int resizeHeight, int resize_quality)
{
    // Decode, Encoder format
    nvjpegOutputFormat_t oformat = NVJPEG_OUTPUT_BGR;
    nvjpegInputFormat_t iformat = NVJPEG_INPUT_BGR;

    // timing for resize
    time = 0.;
    float resize_time = 0.;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 获取文件名字，不带路径和扩展名
    size_t position = sImagePath.rfind("/");
    std::string sFileName = (std::string::npos == position)? sImagePath : sImagePath.substr(position + 1, sImagePath.size());
    position = sFileName.rfind(".");
    sFileName = (std::string::npos == position)? sFileName : sFileName.substr(0, position);

    // Read an image from disk.
    std::ifstream oInputStream(sImagePath.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
    if(!(oInputStream.is_open()))
    {
        std::cerr << "Cannot open image: " << sImagePath << std::endl;
        return EXIT_FAILURE;
    }

    // 获取文件大小，并设置流位置到文件开头
    std::streamsize nSize = oInputStream.tellg();
    oInputStream.seekg(0, std::ios::beg);

    unsigned char * pBuffer = NULL; 
    unsigned char * pResizeBuffer = NULL;
    
    std::vector<char> vBuffer(nSize);
    if (oInputStream.read(vBuffer.data(), nSize))
    {            
        unsigned char * dpImage = (unsigned char *)vBuffer.data();
        
        int nComponent = 0;
        /**
         * 枚举类型 nvjpegChromaSubsampling_t 定义了 JPEG 图像的色度子采样格式
         *     - NVJPEG_CSS_444：4:4:4 色度子采样，表示每个像素都有完整的色度信息
         *     - NVJPEG_CSS_422：4:2:2 色度子采样，表示水平方向上每两个像素共享一个色度样本
         */
        nvjpegChromaSubsampling_t subsampling;
        int widths[NVJPEG_MAX_COMPONENT];  // NVJPEG_MAX_COMPONENT 一般为4，表示图像的最大通道数（例如，RGBA有4个通道）
        int heights[NVJPEG_MAX_COMPONENT];
        int nReturnCode = 0;

        /**
         * nvjpegStatus_t nvjpegGetImageInfo(
         *     nvjpegHandle_t handle,
         *     const unsigned char* data,
         *     size_t size,
         *     int* nComponent,
         *     nvjpegChromaSubsampling_t* subsampling,
         *     int* widths,
         *     int* heights
         * );
         * @brief 获取 JPEG 图像的信息，包括通道数、色度子采样格式以及每个通道的宽度和高度
         * @param handle nvJPEG 句柄
         * @param data 指向 JPEG 图像数据的指针
         * @param size 图像数据的大小（以字节为单位）
         * @param nComponent 输出参数，返回图像的通道数（例如，3表示RGB，4表示RGBA）
         * @param subsampling 输出参数，返回图像的色度子采样格式
         * @param widths 输出参数，返回每个通道的宽度
         * @param heights 输出参数，返回每个通道的高度
         * @return nvjpegStatus_t，表示函数执行的状态
         */
        if (NVJPEG_STATUS_SUCCESS != nvjpegGetImageInfo(nvjpeg_handle, dpImage, nSize, &nComponent, &subsampling, widths, heights))
        {
            std::cerr << "Error decoding JPEG header: " << sImagePath << std::endl;
            return EXIT_FAILURE;
        }
        std::cout << "Image Info: "
                  << " nComponent=" << nComponent
                  << " subsampling=" << subsamplingToString(static_cast<int>(subsampling))
                  << " width[0]=" << widths[0]
                  << " height[0]=" << heights[0]
                  << " width[1]=" << widths[1]
                  << " height[1]=" << heights[1]
                  << " width[2]=" << widths[2]
                  << " height[2]=" << heights[2]
                  << std::endl;

        if(resizeWidth == 0 || resizeHeight == 0)
        {
            resizeWidth = widths[0]/2;
            resizeHeight = heights[0]/2;
        }

        size_t pitchDesc, pitchResize;
        NppiSize srcSize = { (int)widths[0], (int)heights[0] };
        NppiRect srcRoi = { 0, 0, srcSize.width, srcSize.height };
        NppiSize dstSize = { (int)resizeWidth, (int)resizeHeight };
        NppiRect dstRoi = { 0, 0, dstSize.width, dstSize.height };
        NppStatus st;

        /** typedef struct
         * {
         *     cudaStream_t hStream;
         *     int nCudaDeviceId; // From cudaGetDevice()
         *     int nMultiProcessorCount; //From cudaGetDeviceProperties() 
         *     int nMaxThreadsPerMultiProcessor; // From cudaGetDeviceProperties() 
         *     int nMaxThreadsPerBlock; // From cudaGetDeviceProperties()                                                               
         *     size_t nSharedMemPerBlock; // From cudaGetDeviceProperties                                                              
         *     int nCudaDevAttrComputeCapabilityMajor; // From cudaGetDeviceAttribute()                                          
         *     int nCudaDevAttrComputeCapabilityMinor; // From cudaGetDeviceAttribute()                                                 
         *     unsigned int nStreamFlags; // From cudaStreamGetFlags() 
         *     int nReserved0;
         * } NppStreamContext;
         */
        NppStreamContext nppStreamCtx;
        nppStreamCtx.hStream = NULL; // default stream

        nvjpegImage_t imgDesc;
        nvjpegImage_t imgResize;

        if (is_interleaved(oformat))
        {
            pitchDesc = NVJPEG_MAX_COMPONENT * widths[0];
            pitchResize = NVJPEG_MAX_COMPONENT * resizeWidth;
        }
        else
        {
            pitchDesc = 3 * widths[0];
            pitchResize = 3 * resizeWidth;
        }

        cudaError_t eCopy = cudaMalloc(&pBuffer, pitchDesc * heights[0]);
        if (cudaSuccess != eCopy)
        {
            std::cerr << "cudaMalloc failed : " << cudaGetErrorString(eCopy) << std::endl;
            return EXIT_FAILURE;
        }
        cudaError_t eCopy1 = cudaMalloc(&pResizeBuffer, pitchResize * resizeHeight);
        if (cudaSuccess != eCopy1)
        {
            std::cerr << "cudaMalloc failed : " << cudaGetErrorString(eCopy) << std::endl;
            return EXIT_FAILURE;
        }


        imgDesc.channel[0] = pBuffer;
        imgDesc.channel[1] = pBuffer + widths[0] * heights[0];
        imgDesc.channel[2] = pBuffer + widths[0] * heights[0] * 2;
        imgDesc.pitch[0] = (unsigned int)(is_interleaved(oformat) ? widths[0] * NVJPEG_MAX_COMPONENT : widths[0]);
        imgDesc.pitch[1] = (unsigned int)widths[0];
        imgDesc.pitch[2] = (unsigned int)widths[0];

        imgResize.channel[0] = pResizeBuffer;
        imgResize.channel[1] = pResizeBuffer + resizeWidth * resizeHeight;
        imgResize.channel[2] = pResizeBuffer + resizeWidth * resizeHeight * 2;
        imgResize.pitch[0] = (unsigned int)(is_interleaved(oformat) ? resizeWidth * NVJPEG_MAX_COMPONENT : resizeWidth);;
        imgResize.pitch[1] = (unsigned int)resizeWidth;
        imgResize.pitch[2] = (unsigned int)resizeWidth;

        if (is_interleaved(oformat))
        {
            imgDesc.channel[3] = pBuffer + widths[0] * heights[0] * 3;
            imgDesc.pitch[3] = (unsigned int)widths[0];
            imgResize.channel[3] = pResizeBuffer + resizeWidth * resizeHeight * 3;
            imgResize.pitch[3] = (unsigned int)resizeWidth;
        }

        // nvJPEG encoder parameter setting
        CHECK_NVJPEG(nvjpegEncoderParamsSetQuality(nvjpeg_encode_params, resize_quality, NULL));

        CHECK_NVJPEG(nvjpegEncoderParamsSetSamplingFactors(nvjpeg_encode_params, subsampling, NULL));

        // Timing start
        CHECK_CUDA(cudaEventRecord(start, 0));

        /**
         * nvjpegStatus_t nvjpegDecode(
         *     nvjpegHandle_t handle,
         *     nvjpegJpegState_t jpeg_state,
         *     const unsigned char* data,
         *     size_t size,
         *     nvjpegOutputFormat_t output_format,
         *     nvjpegImage_t* destination,
         *     cudaStream_t stream
         * );
         * @brief 解码 JPEG 图像数据并将其转换为指定的输出格式
         * @param handle nvJPEG 句柄
         * @param jpeg_state 解码jpeg图像得状态句柄
         * @param data 指向 JPEG 图像数据的指针
         * @param size 图像数据的大小（以字节为单位）
         * @param output_format 指定解码后图像的输出格式
         * @param destination 指向 nvjpegImage_t 结构体的指针，用于存储解码后的图像数据
         * @param stream CUDA 流，用于异步操作
         * @return nvjpegStatus_t，表示函数执行的状态
         */
        nReturnCode = nvjpegDecode(nvjpeg_handle, nvjpeg_decoder_state, dpImage, nSize, oformat, &imgDesc, NULL);
        if(nReturnCode != 0)
        {
            std::cerr << "Error in nvjpegDecode." << nReturnCode << std::endl;
            return EXIT_FAILURE;
        }

        // image resize
        /* Note: this is the simplest resizing function from NPP. */
        if (is_interleaved(oformat))
        {
            st = nppiResize_8u_C3R_Ctx(imgDesc.channel[0], imgDesc.pitch[0], srcSize, srcRoi,
                imgResize.channel[0], imgResize.pitch[0], dstSize, dstRoi, NPPI_INTER_LANCZOS, nppStreamCtx);
        }
        else
        {
            /**
             * 1 channel，8 bit图像得resize
             */
            st = nppiResize_8u_C1R_Ctx(imgDesc.channel[0], imgDesc.pitch[0], srcSize, srcRoi,
                imgResize.channel[0], imgResize.pitch[0], dstSize, dstRoi, NPPI_INTER_LANCZOS, nppStreamCtx);
            st = nppiResize_8u_C1R_Ctx(imgDesc.channel[1], imgDesc.pitch[1], srcSize, srcRoi,
                imgResize.channel[1], imgResize.pitch[1], dstSize, dstRoi, NPPI_INTER_LANCZOS, nppStreamCtx);
            st = nppiResize_8u_C1R_Ctx(imgDesc.channel[2], imgDesc.pitch[2], srcSize, srcRoi,
                imgResize.channel[2], imgResize.pitch[2], dstSize, dstRoi, NPPI_INTER_LANCZOS, nppStreamCtx);
        }

        if (st != NPP_SUCCESS)
        {
            std::cerr << "NPP resize failed : " << st << std::endl;
            return EXIT_FAILURE;
        }

        // encoding the resize data
        CHECK_NVJPEG(nvjpegEncodeImage(nvjpeg_handle,
            nvjpeg_encoder_state,
            nvjpeg_encode_params,
            &imgResize,
            iformat,
            dstSize.width,
            dstSize.height,
            NULL));

        // retrive the encoded bitstream for file writing
        std::vector<unsigned char> obuffer;
        size_t length;
        CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(
            nvjpeg_handle,
            nvjpeg_encoder_state,
            NULL,
            &length,
            NULL));

        obuffer.resize(length);

        CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(
            nvjpeg_handle,
            nvjpeg_encoder_state,
            obuffer.data(),
            &length,
            NULL));

        // Timing stop
        CHECK_CUDA(cudaEventRecord(stop, 0));
        CHECK_CUDA(cudaEventSynchronize(stop));

        // file writing
        std::cout << "Resize-width: " << dstSize.width << " Resize-height: " << dstSize.height << std::endl;
        std::string output_filename = sOutputPath + "/" + sFileName + ".jpg";
        char directory[120];
        char mkdir_cmd[256];
        std::string folder = sOutputPath;
        output_filename = folder + "/"+ sFileName +".jpg";

        sprintf(directory, "%s", folder.c_str());
        sprintf(mkdir_cmd, "mkdir -p %s 2> /dev/null", directory);

        int ret = system(mkdir_cmd);

        std::cout << "Writing JPEG file: " << output_filename << std::endl;
        std::ofstream outputFile(output_filename.c_str(), std::ios::out | std::ios::binary);
        outputFile.write(reinterpret_cast<const char *>(obuffer.data()), static_cast<int>(length));
    }
    // Free memory
    CHECK_CUDA(cudaFree(pBuffer));
    CHECK_CUDA(cudaFree(pResizeBuffer));

    // get timing
    CHECK_CUDA(cudaEventElapsedTime(&resize_time, start, stop));
    time = (double)resize_time;

    return EXIT_SUCCESS;
}

// *****************************************************************************
// parsing the arguments function
// -----------------------------------------------------------------------------
int processArgs(image_resize_params_t param)
{
    std::string sInputPath(param.input_dir);
    std::string sOutputPath(param.output_dir);
    int resizeWidth = param.width;
    int resizeHeight = param.height;
    int resize_quality = param.quality;

    int error_code = 1;

    double total_time = 0., decode_time = 0.;
    int total_images = 0;

    std::vector<std::string> inputFiles;
    inputFiles.push_back(sInputPath);

    for (unsigned int i = 0; i < inputFiles.size(); i++)
    {
        std::string &sFileName = inputFiles[i];
        std::cout << "Processing file: " << sFileName << std::endl;

        int image_error_code = decodeResizeEncodeOneImage(sFileName, sOutputPath, decode_time, resizeWidth, resizeHeight, resize_quality);

        if (image_error_code)
        {
            std::cerr << "Error processing file: " << sFileName << std::endl;
            return image_error_code;
        }
        else
        {
            total_images++;
            total_time += decode_time;
        }
    }

    std::cout << "------------------------------------------------------------- " << std::endl;
    std::cout << "Total images resized: " << total_images << std::endl;
    std::cout << "Total time spent on resizing: " << total_time << " (ms)" << std::endl;
    std::cout << "Avg time/image: " << total_time/total_images << " (ms)" << std::endl;
    std::cout << "------------------------------------------------------------- " << std::endl;
    return EXIT_SUCCESS;
}

// *****************************************************************************
// main image resize function
// -----------------------------------------------------------------------------
int main(int argc, const char *argv[])
{
    image_resize_params_t params;

    params.input_dir = "/mnt/workspace/cgz_workspace/Exercise/camera_example/input/650.jpeg";
    params.output_dir = "/mnt/workspace/cgz_workspace/Exercise/camera_example/output";
    params.quality = 95;
    params.width = 1080;
    params.height = 720;

    nvjpegDevAllocator_t dev_allocator = {&dev_malloc, &dev_free};
    CHECK_NVJPEG(nvjpegCreate(impl, &dev_allocator, &nvjpeg_handle));
    CHECK_NVJPEG(nvjpegJpegStateCreate(nvjpeg_handle, &nvjpeg_decoder_state));

    // create bitstream object
    CHECK_NVJPEG(nvjpegJpegStreamCreate(nvjpeg_handle, &nvjpeg_jpeg_stream));
    CHECK_NVJPEG(nvjpegDecodeParamsCreate(nvjpeg_handle, &nvjpeg_decode_params));
    CHECK_NVJPEG(nvjpegEncoderStateCreate(nvjpeg_handle, &nvjpeg_encoder_state, NULL));
    CHECK_NVJPEG(nvjpegEncoderParamsCreate(nvjpeg_handle, &nvjpeg_encode_params, NULL));

    // 处理图像
    int ret = processArgs(params);

    CHECK_NVJPEG(nvjpegEncoderParamsDestroy(nvjpeg_encode_params));
    CHECK_NVJPEG(nvjpegDecodeParamsDestroy(nvjpeg_decode_params));
    CHECK_NVJPEG(nvjpegEncoderStateDestroy(nvjpeg_encoder_state));
    CHECK_NVJPEG(nvjpegJpegStateDestroy(nvjpeg_decoder_state));
    CHECK_NVJPEG(nvjpegDestroy(nvjpeg_handle));
    
    return 0;
}
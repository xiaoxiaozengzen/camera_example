# Overview

主要探讨nvidia中对图像以及视频流相关得支持

# 图像处理

![AI图像分类处理](./doc/image_classify_process.png)

图像分类的 pipeline 大体可以分为 2 个部分：
* 前处理部分:
  *  图片解码：将压缩格式图片(jpeg,h264)还原成原始像素矩阵H * W * C
  *  Resize: 将图片调整未要求尺寸
  *  Padding：填充，如果图片宽高比例和目标尺寸不一致，先缩放后再用黑边或者指定颜色进行填充
  *  Normalize：归一化，将像素值缩放到指定范围。原始提图片像素一般是1~255(uint8)，而神经网络参数初始化和激活函数通常假设输入在较小范围0-1之间
* DNN 部分: 用训练好的神经网络模型（如分类、检测、分割等）对新数据进行预测，GPU 可极大加速这一过程

# NvSIPL

NvSIPL: (NVIDIA Sensor Input Processing Library)nvidia drive/jetson平台得相机接口库，用于摄像头得采集、控制、数据流管理

# 术语

NvSCI：(Nvidia Safety Communication Interface)：
* 主要用于跨进程、跨组件的数据缓冲区域管理和安全通信
* 提供统一buffer、event、stream等抽象，保证数据流的安全性，隔离性和高效传递
* 适合多进程/多安全域的数据共享和同步，常用于camera、ISP、DLA、GPU
* 不直接操作硬件，而是提供数据通道跟内存管理

NVSIPL：
* 主要用于摄像头、雷达等传感器的采集、控制和数据流管理
* 封装了sensor驱动、ISP pipeline、帧同步、数据格式等转换功能
* 直接面向sensor硬件跟ISP

L4T：(Linux For Tegra)
- 有个专门为Tegra架构设备的linux操作系统，定制的
- 是为Jetson硬件平台定制
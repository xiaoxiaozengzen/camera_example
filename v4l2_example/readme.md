# Overview

V4l2(Video For Linux 2)：Linux内核中视频设备中的驱动框架，对于应用层它提供了一系列的API接口，同时对于硬件层，它适配大部分的视频设备，因此通过调用V4L2的接口函数可以适配大部分的视频设备。

Video设备又分为主设备和从设备，对于Camera来说：
* 主设备：Camera Host控制器为主设备，负责图像数据的接收和传输， 
* 从设备： 从设备为Camera Sensor，一般为I2C接口，可通过从设备控制Camera采集图像的行为，如图像的大小、图像的FPS等。

V4L2的主设备号是81，次设备号范围0~255 这些次设备号又分为多类设备：
* 视频设备（次设备号范围0-63）
* Radio（收音机）设备（次设备号范围64-127）
* Teletext设备（次设备号范围192-223）
* VBI设备（次设备号范围224-255）。

V4L2设备对应的设备节点有**/dev/videoX、/dev/vbiX、/dev/radioX。 其中，视频设备对应的设备节点是/dev/videoX**，视频设备以高频摄像头或Camera为输入源，Linux内核驱动该类设备，接收相应的视频信息并处理。

可以查看当前环境中的相关设备：

```bash
ls /dev/vi*
/dev/video0
/dev/video1

# /dev/videoX: 第X个被系统识别的视频设备，比如：第一个摄像头，采集卡等
# 可以使用命令进行查看当前的video设备：v4l2-ctl --list-devices
```
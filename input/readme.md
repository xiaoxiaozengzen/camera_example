# Overview

一些使用ffmpeg的方式

# 命令

```bash
# 将jpg转成yuv
ffmpeg -i input.jpg -pix_fmt yuv420p output.yuv

# yuv转jpg
# * -f rawvideo: 输入为原始视频流(无文件头)
# * -pix_fmt yuv420p：像素格式
# * -s 1920x1080：指定分辨率 width*height
# * -frames:v 1: 只输出一帧
# 注意：图像的大小一定要输入准确，否则转换后的图像会像素位置错乱
ffmpeg -f rawvideo -pix_fmt yuv420p -s 1920x1080 -i input.yuv -frames:v 1 output.jpg

# mp42yuv： 只转一帧，没有-frames:v 1参数的话，会把所有的yuv数据存到一张图
ffmpeg -i input.mp4 -pix_fmt yuv420p -frames:v 1 output.yuv

# mp42yuv
# * -f segment：使用segment多文件分段器，将输出分割为多个文件
# * -segment_time 0.04: 使用与25fps(1/25=0.04)
# * -reset_timestamps 1：每个分段文件的时间戳从0开始
# * %03d：输出文件格式，表示三位数字编号，按照0补全
ffmpeg -i input.mp4 -pix_fmt yuv420p -f segment -segment_time 0.04 -reset_timestamps 1 frame_%03d.yuv
```
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

# 从MP4提取H.264原始流，并存储在output.h264文件中
# -c:v 视频流的视频编解码选项(codec for video):copy表示逐packet拷贝，不进行编解码
# -bsf:v h264_mp4toannexb：为视频流应用bsf，并指定264_mp4toannexb，将NALU的AVCC格式转Annex-B。如果输入是hevc则需要改成hevc_mp4toannexb
# -an：no audio，去掉音频流
ffmpeg -i input.mp4 -c:v copy -bsf:v h264_mp4toannexb -an output.h264

# 将input.h26中H.264的原始流，封装到output.mp4中
# 找不到参数h264_annexbtomp4，用其他命令代替
ffmpeg -i input.h264 -c:v copy -bsf:v h264_annexbtomp4 -an output.mp4
# 或
ffmpeg -f h264 -i input.h264 -c:v copy output.mp4

```
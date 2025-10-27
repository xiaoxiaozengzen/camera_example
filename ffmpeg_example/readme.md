# Overview

主要叙述h264/h265的编解码知识，以及ffmpeg的相关知识

# H264/h265

## 编码结构

H.265在编码结构上分为视频编码层（VCL）和网络提取层（NAL）：
* VCL：Video Coding Layer，主要包括视频压缩引擎和图像分块的语法定义，原始视频在 VCL 层，被编码成视频数据。简单版本的编码过程如下：
  * 将每一帧的图像分块，将块信息添加到码流中；
  * 对单元块进行预测编码，帧内预测生成残差，帧间预测进行运动估计和运动补偿；
  * 对残差进行变换，对变换系数进行量化、扫描。
  * 对量化后的变换系数、运动信息、预测信息等进行熵编码，形成压缩的视频码流输出。
* NAL：Network Abstraction Layer，主要定义数据的封装格式，把 VCL 产生的视频数据封装成一个个 NAL 单元的数据包，适配不同的网络环境并传输。

## 帧类型

在H264协议里定义了三种帧：
* 完整编码的帧叫I帧，
* 参考之前的I帧生成的只包含差异部分编码的帧叫P帧
* 还有一种参考前后的帧编码的帧叫B帧。

## 压缩方式

H264的压缩方法:
* 分组:把几帧图像分为一组(GOP，也就是一个序列),为防止运动变化,帧数不宜取多。
* 定义帧:将每组内各帧图像定义为三种类型,即I帧、B帧和P帧;
* 预测帧:以I帧做为基础帧,以I帧预测P帧,再由I帧和P帧预测B帧;
* 数据传输:最后将I帧数据与预测的差值信息进行存储和传输。

## 术语

* NAL unit（网络抽象层单元）：
  * H.264 的基本封装单元。每个 NAL 包含一个 header（指示类型和重要性）和 RBSP（原始比特流语法）。
* RBSP（Raw Byte Sequence Payload）
  * NAL header 后的有效比特流部分，包含 slice header、SPS/PPS、SEI 等内容（经防止 start-code 出现的字节插入处理）。
* Access Unit（AU） / coded picture（编码图像）
  * 构成“编码一帧”的 NAL 集合（包含所有 slice 和可选辅助 NAL）。送给解码器的单位通常是 AU。
  * AU边界(起始跟结束)的判定很富在，一般工程上：demuxer(mp4)/parser(libavcodes的av_parser)来正确组装AU
* VCL（Video Coding Layer）与 non‑VCL：
  * VCL NAL（如 slice）承载实际像素编码；
  * non‑VCL（如 SPS/PPS/SEI/AUD）是参数或元数据。
* SPS / PPS（序列参数集 / 图像参数集）
  * SPS（nal_type=7）和 PPS（nal_type=8）包含解码所需的重要参数（分辨率、profile、色彩、帧/场配置等）。常作为 extradata 存放或随流传输。
* Slice（切片）
  * 一帧可以分成多个 slice，每个 slice 包含若干宏块；slice 允许并行/容错。
* Macroblock / transform / motion vectors
  * 编码的像素单元、变换系数、运动估计等构成压缩过程基本计算单元（现代实现还有 4x4、8x8 变体等）。
* I/P/B 帧（关键帧 / 预测帧 / 双向预测帧）
  * I（Intra）自我编码、
  * P 参考之前帧、
  * B 参考前/后帧。B 帧引入重排（DTS vs PTS）。
* POC（Picture Order Count）
  * 表示显示顺序的计数器，用于重排与参考管理。
* DPB（Decoded Picture Buffer）
  * 解码器保留的已解码参考帧集合，用于预测/重用。
* CABAC / CAVLC：
  * 熵编码方法：CABAC 更高效但更复杂，CAVLC 更简单。由 profile 决定是否能用 CABAC。
* SEI（Supplemental Enhancement Information）
  * 附加信息 NAL（nal_type=6），非必需但可携带重要元数据（字幕、时间码、HDR 元数据等）。
* AUD（Access Unit Delimiter，nal_type=9）
  * 可选，标识 AU 边界的辅助 NAL（并不总存在）
# Overview

nsys：Nsight Systems 是 NVIDIA 提供的一款性能分析工具，主要用于分析 GPU 和 CPU 之间的工作负载，以帮助开发者找到瓶颈并优化性能。它可以提供从 CPU 调度到 GPU 内核执行的详细时序视图，帮助识别性能瓶颈。

# 下载

```bash
wget https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2025_6/nsight-systems-2025.6.1_2025.6.1.190-1_amd64.deb
```

# 使用

## 基本使用

```bash
# 这个命令会生成一个 .qdrep或者.nsys-rep 格式的报告文件，可以使用GUI工具进行分析
nsys profile --stats=true -o report_name ./your_program
# profile 是 Nsight Systems 的主要命令，表示进行性能分析。
# --stats=true 表示在分析后打印统计信息。
# -o report_name 表示生成的报告文件名。
# ./your_program 是要分析的可执行程序。
```

## 可视化

```bash
nsys-ui <your_report_path>.nsys-rep
```
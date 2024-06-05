# Basic Demo

Read this in [English](README_en.md)

本 demo 中，你将体验到如何使用 glm-4-9b 开源模型进行基本的任务。

请严格按照文档的步骤进行操作，以避免不必要的错误。

## 设备和依赖检查

### 相关推理测试数据

**本文档的数据均在以下硬件环境测试,实际运行环境需求和运行占用的显存略有不同，请以实际运行环境为准。**
测试硬件信息:

+ OS: Ubuntu 22.04
+ Memory: 512GB
+ Python: 3.12.3
+ CUDA Version:  12.3
+ GPU Driver: 535.104.05
+ GPU: NVIDIA A100-SXM4-80GB * 8

相关推理的压力测试数据如下：

**所有测试均在单张GPU上进行测试,所有显存消耗都按照峰值左右进行测算**

| 精度   | 显存占用     | Prefilling / 首响 | Decode Speed     | Remarks      |
|------|----------|-----------------|------------------|--------------|
| BF16 | 19047MiB | 0.1554s         | 27.8193 tokens/s | 输入长度为 1000   |
| BF16 | 20629MiB | 0.8199s         | 31.8613 tokens/s | 输入长度为 8000   |
| BF16 | 27779MiB | 4.3554s         | 14.4108 tokens/s | 输入长度为 32000  |
| BF16 | 57379MiB | 38.1467s        | 3.4205  tokens/s | 输入长度为 128000 |
| BF16 | 74497MiB | 98.4930s        | 2.3653  tokens/s | 输入长度为 200000 |

| 精度   | 显存占用     | Prefilling / 首响 | Decode Speed     | Remarks     |
|------|----------|-----------------|------------------|-------------|
| Int4 | 8251MiB  | 0.1667s         | 23.3903 tokens/s | 输入长度为 1000  |
| Int4 | 9613MiB  | 0.8629s         | 23.4248 tokens/s | 输入长度为 8000  |
| Int4 | 16065MiB | 4.3906s         | 14.6553 tokens/s | 输入长度为 32000 |

### 最低硬件要求

如果您希望运行官方提供的最基础代码 (transformers 后端) 您需要：

+ Python >= 3.10
+ 内存不少于 32 GB

如果您希望运行官方提供的本文件夹的所有代码，您还需要：

+ Linux 操作系统 (Debian 系列最佳)
+ 大于 8GB 显存的，支持 CUDA 或者 ROCM 并且支持 `BF16` 推理的 GPU 设备 (A100以上GPU，V100，20以及更老的GPU架构不受支持)

安装依赖

```shell
pip install -r requirements.txt
```

## 基础功能调用

**除非特殊说明，本文件夹所有 demo 并不支持 Function Call 和 All Tools 等进阶用法**

### 使用 transformers 后端代码

+ 使用 命令行 与 glm-4-9b 模型进行对话。

```shell
python trans_cli_demo.py
```

+ 使用 Gradio 网页端与 glm-4-9b 模型进行对话。

```shell
python trans_web_demo.py
```

+ 使用 Batch 推理。

```shell
python cli_batch_request_demo.py
```

### 使用 VLLM 后端代码

+ 使用命令行与 glm-4-9b 模型进行对话。

```shell
python vllm_cli_demo.py
```

+ 自行构建服务端，并使用 `OpenAI API` 的请求格式与 glm-4-9b 模型进行对话。本 demo 支持 Function Call 和 All Tools功能。

启动服务端：

```shell
python openai_api_server.py
```

客户端请求：

```shell
python openai_api_request.py
```

## 压力测试

用户可以在自己的设备上使用本代码测试模型在 transformers后端的生成速度:

```shell
python trans_stress_test.py
```




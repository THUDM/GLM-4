# Inference

Read this in [English](README.md)

请严格按照文档的步骤进行操作，以避免不必要的错误。

## 设备和依赖检查

### 安装依赖

```shell
pip install -r requirements.txt
```

### 相关推理测试数据

**本文档的数据均在以下硬件环境测试,实际运行环境需求和运行占用的显存略有不同，请以实际运行环境为准。**

测试硬件信息:

+ OS: Ubuntu 22.04
+ Memory: 512GB
+ Python: 3.12.3
+ CUDA Version:  12.4
+ Cmake 3.23.0
+ GPU Driver: 535.104.05
+ GPU: NVIDIA H100 80GB HBM3 * 8

推理的压力测试数据如下，如有多张显卡，则显存占用代表显存占用最大一张显卡的显存消耗。

#### GLM-4-32B-0414

| 精度   | 显卡数量 | 显存占用  | 首 Token 延迟 | Token 输出速度    | 输入token数 |
|------|------|-------|------------|---------------|----------|
| BF16 | 1    | 68 GB | 0.16s      | 24.4 tokens/s | 1000     |
| BF16 | 1    | 72 GB | 1.37s      | 16.9 tokens/s | 8000     |
| BF16 | 2    | 50 GB | 6.75s      | 8.1 tokens/s  | 32000    |
| BF16 | 4    | 55 GB | 37.83s     | 3.0 tokens/s  | 100000   |

#### GLM-4-9B-0414

| 精度   | 显卡数量 | 显存占用  | 首 Token 延迟 | Token 输出速度    | 输入token数 |
|------|------|-------|------------|---------------|---------|
| BF16 | 1    | 19 GB | 0.05s      | 44.4 tokens/s | 1000    |
| BF16 | 1    | 25 GB | 0.39s      | 39.0 tokens/s | 8000    |
| BF16 | 1    | 31 GB | 2.29s      | 18.7 tokens/s | 32000   |
| BF16 | 1    | 55 GB | 6.80s      | 14.1 tokens/s  | 100000  |


#### GLM-4-9B-Chat-1M

| 精度     | 显卡数量 | 显存占用  | 首 Token 延迟 | Token 输出速度    | 输入token数 |
|--------|------|------|------------|--------------|-------------|
| BF16 | 1    | 75 GB | 98.4s      | 2.3 tokens/s | 200000 |

#### GLM-4V-9B

| 精度     | 显卡数量 | 显存占用  | 首 Token 延迟 | Token 输出速度    | 输入token数 |
|--------|------|------|------------|--------------|-------------|
| BF16 | 1    | 28 GB | 0.1s       | 33.4 tokens/s | 1000 |
| BF16 | 1    | 33 GB | 0.7s       | 39.2 tokens/s | 8000 |

| 精度     | 显卡数量  | 显存占用   | 首 Token 延迟 | Token 输出速度    | 输入token数 |
|--------|-------|--------|------------|--------------|-------------|
| INT4 | 1     | 10 GB  | 0.1s       | 28.7 tokens/s |  1000 |
| INT4 | 1     | 15 GB  | 0.8s       | 24.2 tokens/s |  8000 |

## 快速开始

### 使用 transformers 后端代码

+ 使用命令行与 GLM-4-9B 模型进行对话。

```shell
python trans_cli_demo.py # LLM Such as GLM-4-9B-0414
python trans_cli_vision_demo.py # GLM-4V-9B
```

+ 使用 Gradio 网页端与 GLM-4-9B 模型进行对话。

```shell
python trans_web_demo.py  # LLM Such as GLM-4-9B-0414
python trans_web_vision_demo.py # GLM-4V-9B
```

+ 使用 Batch 推理。

```shell
python trans_batch_demo.py
```

### 使用 vLLM 后端代码

+ 使用命令行与 GLM-4-9B-Chat 模型进行对话。

```shell
python vllm_cli_demo.py # LLM Such as GLM-4-9B-0414
```

+ 构建 OpenAI 类 API 服务。
```shell
vllm serve THUDM/GLM-4-9B-0414 --tensor_parallel_size 2
```

### 使用 glm-4v 构建 OpenAI 服务

启动服务端

```shell
python glm4v_server.py THUDM/glm-4v-9b
```

客户端请求：

```shell
python glm4v_api_request.py
```

## 压力测试

用户可以在自己的设备上使用本代码测试模型在 transformers后端的生成速度:

```shell
python trans_stress_test.py
```

## 压力测试

用户可以在自己的设备上使用本代码测试模型在 transformers后端的生成速度:

```shell
python trans_stress_test.py
```

压力测试脚本支持开启**SwanLab**来跟踪压力测试过程和记录指标：

```shell
# API Key 可通过登录https://swanlab.cn/获取
python trans_stress_test.py --swanlab_api_key "SwanLab的API Key"

```
使用`--swanlab_api_key local`参数可开启SwanLab本地模式

## 使用昇腾NPU运行代码

用户可以在昇腾硬件环境下运行以上代码，只需将transformers修改为openmind，将device中的cuda设备修改为npu：

```shell
#from transformers import AutoModelForCausalLM, AutoTokenizer
from openmind import AutoModelForCausalLM, AutoTokenizer

#device = 'cuda'
device = 'npu'
```

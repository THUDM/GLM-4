# Inference

[中文阅读](README_zh.md)

Please follow the steps in the document strictly to avoid unnecessary errors.

## Device and dependency check

### Install dependencies

```shell
pip install -r requirements.txt
```

### Related Inference Benchmark Data

**All benchmark data in this document was collected under the hardware environment listed below. Actual memory usage and runtime may vary depending on your deployment setup. Please refer to your actual environment.**

Test Hardware:

+ OS: Ubuntu 22.04
+ Memory: 512GB
+ Python: 3.12.3
+ Cmake 3.23.0
+ CUDA Version: 12.4
+ GPU Driver: 535.104.05
+ GPU: NVIDIA H100 80GB HBM3 * 8

The following stress test results show memory usage and latency during inference. If multiple GPUs are used, "Memory Usage" refers to the maximum usage on a single GPU.

#### GLM-4-32B-Chat-0414

| Precision   | #GPUs | Memory Usage  | First Token Latency | Token Output Speed | Input Tokens |
|-------------|-------|---------------|---------------------|-------------------|--------------|
| BF16        | 1     | 68 GB         | 0.16s               | 24.4 tokens/s     | 1000         |
| BF16        | 1     | 72 GB         | 1.37s               | 16.9 tokens/s     | 8000         |
| BF16        | 2     | 50 GB         | 6.75s               | 8.1 tokens/s      | 32000        |
| BF16        | 4     | 55 GB         | 37.83s              | 3.0 tokens/s      | 100000       |

#### GLM-4-9B-Chat-0414

| Precision | #GPUs | Memory Usage | First Token Latency | Token Output Speed | Input Tokens |
|-----------|-------|---------------|----------------------|---------------------|---------------|
| BF16      | 1     | 19 GB         | 0.05s                | 44.4 tokens/s       | 1000          |
| BF16      | 1     | 25 GB         | 0.39s                | 39.0 tokens/s       | 8000          |
| BF16      | 1     | 31 GB         | 2.29s                | 18.7 tokens/s       | 32000         |
| BF16      | 1     | 55 GB         | 6.80s                | 14.1 tokens/s       | 100000        |

#### GLM-4-9B-Chat-1M

| Precision | #GPUs | Memory Usage | First Token Latency | Token Output Speed | Input Tokens |
|-----------|-------|---------------|----------------------|---------------------|---------------|
| BF16      | 1     | 75 GB         | 98.4s                | 2.3 tokens/s        | 200000        |

#### GLM-4V-9B

| Precision | #GPUs | Memory Usage | First Token Latency | Token Output Speed | Input Tokens |
|-----------|-------|---------------|----------------------|---------------------|---------------|
| BF16      | 1     | 28 GB         | 0.1s                 | 33.4 tokens/s       | 1000          |
| BF16      | 1     | 33 GB         | 0.7s                 | 39.2 tokens/s       | 8000          |

| Precision | #GPUs | Memory Usage | First Token Latency | Token Output Speed | Input Tokens |
|-----------|-------|---------------|----------------------|---------------------|---------------|
| INT4      | 1     | 10 GB         | 0.1s                 | 28.7 tokens/s       | 1000          |
| INT4      | 1     | 15 GB         | 0.8s                 | 24.2 tokens/s       | 8000          |

## Quick Start

### Use transformers backend code

+ Use the command line to communicate with the GLM-4-9B model.

```shell
python trans_cli_demo.py # LLM Such as GLM-4-9B-Chat-0414
python trans_cli_vision_demo.py # GLM-4V-9B
```

+ Use the Gradio web client to communicate with the  GLM-4-9B model.

```shell
python trans_web_demo.py  # LLM Such as GLM-4-9B-Chat-0414
python trans_web_vision_demo.py # GLM-4V-9B
```

+ Use Batch inference.

```shell
python trans_batch_demo.py  # LLM Such as GLM-4-9B-Chat-0414
```

### Use vLLM backend code

+ Use the command line to communicate with the GLM-4-9B-Chat model.

```shell
python vllm_cli_demo.py  # LLM Such as GLM-4-9B-Chat-0414
```

+ Launch an OpenAI-compatible API service.

```shell
vllm serve THUDM/GLM-4-9B-Chat-0414 --tensor_parallel_size 2
```

### Use glm-4v to build an OpenAI-compatible service

Start the server:

```shell
python glm4v_server.py THUDM/glm-4v-9b
```

Client request:

```shell
python glm4v_api_request.py
```

## Stress test

Users can use this code to test the generation speed of the model on the transformers backend on their own devices:

```shell
python trans_stress_test.py
```

The stress test script supports enabling **SwanLab** to track the stress testing process and record metrics:

```shell
# The API Key can be obtained by logging in to https://swanlab.cn/
python trans_stress_test.py --swanlab_api_key "Your SwanLab API Key"
```

Using the --swanlab_api_key local parameter enables SwanLab's local mode.

## Use Ascend card to run code

Users can run the above code in the Ascend hardware environment. They only need to change the transformers to openmind and the cuda device in device to npu.

```shell
#from transformers import AutoModelForCausalLM, AutoTokenizer
from openmind import AutoModelForCausalLM, AutoTokenizer

#device = 'cuda'
device = 'npu'
```

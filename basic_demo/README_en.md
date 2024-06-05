# Basic Demo

In this demo, you will experience how to use the glm-4-9b open source model to perform basic tasks.

Please follow the steps in the document strictly to avoid unnecessary errors.

## Device and dependency check

### Related inference test data

**The data in this document are tested in the following hardware environment. The actual operating environment
requirements and the video memory occupied by the operation are slightly different. Please refer to the actual operating
environment. **
Test hardware information:

+ OS: Ubuntu 22.04
+ Memory: 512GB
+ Python: 3.12.3
+ CUDA Version: 12.3
+ GPU Driver: 535.104.05
+ GPU: NVIDIA A100-SXM4-80GB * 8

The stress test data of relevant inference are as follows:

**All tests are performed on a single GPU, and all video memory consumption is calculated based on the peak value**

#

### GLM-4-9B-Chat

| Dtype | GPU Memory | Prefilling | Decode Speed     | Remarks                |
|-------|------------|------------|------------------|------------------------|
| BF16  | 19047MiB   | 0.1554s    | 27.8193 tokens/s | Input length is 1000   |
| BF16  | 20629MiB   | 0.8199s    | 31.8613 tokens/s | Input length is 8000   |
| BF16  | 27779MiB   | 4.3554s    | 14.4108 tokens/s | Input length is 32000  |
| BF16  | 57379MiB   | 38.1467s   | 3.4205  tokens/s | Input length is 128000 |

| Dtype | GPU Memory | Prefilling | Decode Speed     | Remarks               |
|-------|------------|------------|------------------|-----------------------|
| Int4  | 8251MiB    | 0.1667s    | 23.3903 tokens/s | Input length is 1000  |
| Int4  | 9613MiB    | 0.8629s    | 23.4248 tokens/s | Input length is 8000  |
| Int4  | 16065MiB   | 4.3906s    | 14.6553 tokens/s | Input length is 32000 |

### GLM-4-9B-Chat-1M

| Dtype | GPU Memory | Prefilling | Decode Speed     | Remarks      |
|-------|------------|------------|------------------|--------------|
| BF16  | 74497MiB   | 98.4930s   | 2.3653  tokens/s | 输入长度为 200000 |

If your input exceeds 200K, we recommend that you use the VLLM backend with multi gpus for inference to get better performance.

#### GLM-4V-9B

| Dtype | GPU Memory | Prefilling | Decode Speed     | Remarks              |
|-------|------------|------------|------------------|----------------------|
| BF16  | 28131MiB   | 0.1016s    | 33.4660 tokens/s | Input length is 1000 |
| BF16  | 33043MiB   | 0.7935a    | 39.2444 tokens/s | Input length is 8000 |

| Dtype | GPU Memory | Prefilling | Decode Speed     | Remarks              |
|-------|------------|------------|------------------|----------------------|
| Int4  | 10267MiB   | 0.1685a    | 28.7101 tokens/s | Input length is 1000 |
| Int4  | 14105MiB   | 0.8629s    | 24.2370 tokens/s | Input length is 8000 |

### Minimum hardware requirements

If you want to run the most basic code provided by the official (transformers backend) you need:

+ Python >= 3.10
+ Memory of at least 32 GB

If you want to run all the codes in this folder provided by the official, you also need:

+ Linux operating system (Debian series is best)
+ GPU device with more than 8GB video memory, supporting CUDA or ROCM and supporting `BF16` reasoning (GPUs above A100,
  V100, 20 and older GPU architectures are not supported)

Install dependencies

```shell
pip install -r requirements.txt
```

## Basic function calls

**Unless otherwise specified, all demos in this folder do not support advanced usage such as Function Call and All Tools
**

### Use transformers backend code

+ Use the command line to communicate with the glm-4-9b model.

```shell
python trans_cli_demo.py
```

+ Use the Gradio web client to communicate with the glm-4-9b model.

```shell
python trans_web_demo.py
```

+ Use Batch inference.

```shell
python cli_batch_request_demo.py
```

### Use VLLM backend code

+ Use the command line to communicate with the glm-4-9b model.

```shell
python vllm_cli_demo.py
```

+ Build the server by yourself and use the request format of `OpenAI API` to communicate with the glm-4-9b model. This
  demo supports Function Call and All Tools functions.

Start the server:

```shell
python openai_api_server.py
```

Client request:

```shell
python openai_api_request.py
```

## Stress test

Users can use this code to test the generation speed of the model on the transformers backend on their own devices:

```shell
python trans_stress_test.py
```
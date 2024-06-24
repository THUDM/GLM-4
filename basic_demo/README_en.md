# Basic Demo

In this demo, you will experience how to use the GLM-4-9B open source model to perform basic tasks.

Please follow the steps in the document strictly to avoid unnecessary errors.

## Device and dependency check

### Related inference test data

**The data in this document are tested in the following hardware environment. The actual operating environment
requirements and the GPU memory occupied by the operation are slightly different. Please refer to the actual operating
environment.**

Test hardware information:

+ OS: Ubuntu 22.04
+ Memory: 512GB
+ Python: 3.10.12 (recommend) / 3.12.3 have been tested
+ CUDA Version: 12.3
+ GPU Driver: 535.104.05
+ GPU: NVIDIA A100-SXM4-80GB * 8

The stress test data of relevant inference are as follows:

**All tests are performed on a single GPU, and all GPU memory consumption is calculated based on the peak value**

#

### GLM-4-9B-Chat

| Dtype | GPU Memory | Prefilling | Decode Speed  | Remarks                |
|-------|------------|------------|---------------|------------------------|
| BF16  | 19 GB      | 0.2s       | 27.8 tokens/s | Input length is 1000   |
| BF16  | 21 GB      | 0.8s       | 31.8 tokens/s | Input length is 8000   |
| BF16  | 28 GB      | 4.3s       | 14.4 tokens/s | Input length is 32000  |
| BF16  | 58 GB      | 38.1s      | 3.4  tokens/s | Input length is 128000 |

| Dtype | GPU Memory | Prefilling | Decode Speed  | Remarks               |
|-------|------------|------------|---------------|-----------------------|
| INT4  | 8 GB       | 0.2s       | 23.3 tokens/s | Input length is 1000  |
| INT4  | 10 GB      | 0.8s       | 23.4 tokens/s | Input length is 8000  |
| INT4  | 17 GB      | 4.3s       | 14.6 tokens/s | Input length is 32000 |

### GLM-4-9B-Chat-1M

| Dtype | GPU Memory | Prefilling | Decode Speed     | Remarks                |
|-------|------------|------------|------------------|------------------------|
| BF16  | 74497MiB   | 98.4s      | 2.3653  tokens/s | Input length is 200000 |

If your input exceeds 200K, we recommend that you use the vLLM backend with multi gpus for inference to get better
performance.

#### GLM-4V-9B

| Dtype | GPU Memory | Prefilling | Decode Speed  | Remarks              |
|-------|------------|------------|---------------|----------------------|
| BF16  | 28 GB      | 0.1s       | 33.4 tokens/s | Input length is 1000 |
| BF16  | 33 GB      | 0.7s       | 39.2 tokens/s | Input length is 8000 |

| Dtype | GPU Memory | Prefilling | Decode Speed  | Remarks              |
|-------|------------|------------|---------------|----------------------|
| INT4  | 10 GB      | 0.1s       | 28.7 tokens/s | Input length is 1000 |
| INT4  | 15 GB      | 0.8s       | 24.2 tokens/s | Input length is 8000 |

### Minimum hardware requirements

If you want to run the most basic code provided by the official (transformers backend) you need:

+ Python >= 3.10
+ Memory of at least 32 GB

If you want to run all the codes in this folder provided by the official, you also need:

+ Linux operating system (Debian series is best)
+ GPU device with more than 8GB GPU memory, supporting CUDA or ROCM and supporting `BF16` reasoning (`FP16` precision
  cannot be finetuned, and there is a small probability of problems in infering)

Install dependencies

```shell
pip install -r requirements.txt
```

## Basic function calls

**Unless otherwise specified, all demos in this folder do not support advanced usage such as Function Call and All Tools
**

### Use transformers backend code

+ Use the command line to communicate with the GLM-4-9B model.

```shell
python trans_cli_demo.py # GLM-4-9B-Chat
python trans_cli_vision_demo.py # GLM-4V-9B
```

+ Use the Gradio web client to communicate with the  GLM-4-9B model.

```shell
python trans_web_demo.py  # GLM-4-9B-Chat
python trans_web_vision_demo.py # GLM-4V-9B
```

+ Use Batch inference.

```shell
python trans_batch_demo.py
```

### Use vLLM backend code

+ Use the command line to communicate with the GLM-4-9B-Chat model.

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
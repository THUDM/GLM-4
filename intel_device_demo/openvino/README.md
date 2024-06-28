# 使用 OpenVINO 部署 GLM-4-9B-Chat 模型

Read this in [English](README_en.md).

[OpenVINO](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html) 
是 Intel 为深度学习推理而设计的开源工具包。它可以帮助开发者优化模型，提高推理性能，减少模型的内存占用。
本示例将展示如何使用 OpenVINO 部署 GLM-4-9B-Chat 模型。

## 1. 环境配置

首先，你需要安装依赖

```bash
pip install -r requirements.txt
```

## 2. 转换模型

由于需要将Huggingface模型转换为OpenVINO IR模型，因此您需要下载模型并转换。

```
python3 convert.py --model_id THUDM/glm-4-9b-chat --output {your_path}/glm-4-9b-chat-ov
```

### 可以选择的参数

* `--model_id` - 模型所在目录的路径（绝对路径）。
* `--output` - 转换后模型保存的地址。
* `--precision` - 转换的精度。


转换过程如下：
```
====Exporting IR=====
Framework not specified. Using pt to export the model.
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:04<00:00,  2.14it/s]
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Using framework PyTorch: 2.3.1+cu121
Mixed-Precision assignment ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 160/160 • 0:01:45 • 0:00:00
INFO:nncf:Statistics of the bitwidth distribution:
┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
│   Num bits (N) │ % all parameters (layers)   │ % ratio-defining parameters (layers)   │
┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
│              8 │ 31% (76 / 163)              │ 20% (73 / 160)                         │
├────────────────┼─────────────────────────────┼────────────────────────────────────────┤
│              4 │ 69% (87 / 163)              │ 80% (87 / 160)                         │
┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙
Applying Weight Compression ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 163/163 • 0:03:46 • 0:00:00
Configuration saved in glm-4-9b-ov/openvino_config.json
====Exporting tokenizer=====
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
```
## 3. 运行 GLM-4-9B-Chat 模型

```
python3 chat.py --model_path {your_path}/glm-4-9b-chat-ov --max_sequence_length 4096 --device CPU
```

### 可以选择的参数

* `--model_path` - OpenVINO IR 模型所在目录的路径。
* `--max_sequence_length` - 输出标记的最大大小。
* `--device` - 运行推理的设备。

### 参考代码

本代码参考 [OpenVINO 官方示例](https://github.com/OpenVINO-dev-contest/chatglm3.openvino) 进行修改。
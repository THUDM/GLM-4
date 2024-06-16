# GLM-4-9B Chat 对话模型微调

Read this in [English](README_en.md)

本 demo 中，你将体验到如何微调 GLM-4-9B-Chat 对话开源模型(不支持视觉理解模型)。 请严格按照文档的步骤进行操作，以避免不必要的错误。

## 硬件检查

**本文档的数据均在以下硬件环境测试,实际运行环境需求和运行占用的显存略有不同，请以实际运行环境为准。**
测试硬件信息:

+ OS: Ubuntu 22.04
+ Memory: 512GB
+ Python: 3.10.12 / 3.12.3 (如果您使用 Python 3.12.3 目前需要使用 git 源码安装 nltk)
+ CUDA Version:  12.3
+ GPU Driver: 535.104.05
+ GPU: NVIDIA A100-SXM4-80GB * 8

| 微调方案               | 显存占用                              | 权重保存点大小 |
|--------------------|-----------------------------------|---------|
| lora (PEFT)        | 21531MiB                          | 17M     |
| p-tuning v2 (PEFT) | 21381MiB                          | 121M    |
| SFT (Zero3 method) | 80935MiB<br/>(Each GPU，需要使用8张GPU) | 20G     |

在开始微调之前，请你先安装`basic_demo`中的依赖，同时您需要安装本目录下的依赖项：

> NOTE: NLTK 3.8.1 部分代码可能尚未对 python 3.12
> 进行适配，该情况下的适配方法可参考[issues #38](https://github.com/THUDM/GLM-4/issues/38)

```bash
pip install -r requirements.txt
```

## 多轮对话格式

多轮对话微调示例采用 GLM-4 对话格式约定，对不同角色添加不同 `loss_mask` 从而在一遍计算中为多轮回复计算 `loss`。

对于数据文件，样例采用如下格式

如果您仅希望微调模型的对话能力，而非工具能力，您应该按照以下格式整理数据。

```json
[
  {
    "messages": [
      {
        "role": "system",
        "content": "<system prompt text>",
        "tools": [
          {
            "name": "<tool name>",
            "args": {
              "<arg name>": "<arg value>"
            }
          }
          // Add more tools if needed
        ]
      },
      {
        "role": "user",
        "content": "<user prompt text>"
      },
      {
        "role": "assistant",
        "content": "<assistant response text>"
      },
      // If Tool Using
      {
        "role": "user",
        "content": "<user prompt text>"
      },
      {
        "role": "assistant",
        "content": "<assistant response text>"
      },
      {
        "role": "observation",
        "content": "<observation prompt text>"
      },
      {
        "role": "assistant",
        "content": "<assistant response observation>"
      },
      // Multi_turns
      {
        "role": "user",
        "content": "<user prompt text>"
      },
      {
        "role": "assistant",
        "content": "<assistant response text>"
      }
    ]
  }
]
```

这里是一个不带有工具的例子:

```
{"messages": [{"role": "user", "content": "类型#裤*材质#牛仔布*风格#性感"}, {"role": "assistant", "content": "3x1的这款牛仔裤采用浅白的牛仔面料为裤身材质，其柔然的手感和细腻的质地，在穿着舒适的同时，透露着清纯甜美的个性气质。除此之外，流畅的裤身剪裁将性感的腿部曲线彰显的淋漓尽致，不失为一款随性出街的必备单品。"}]}
```

这是一个带有工具调用的例子:

```
{"messages": [{"role": "system", "content": "", "tools": [{"type": "function", "function": {"name": "get_recommended_books", "description": "Get recommended books based on user's interests", "parameters": {"type": "object", "properties": {"interests": {"type": "array", "items": {"type": "string"}, "description": "The interests to recommend books for"}}, "required": ["interests"]}}}]}, {"role": "user", "content": "Hi, I am looking for some book recommendations. I am interested in history and science fiction."}, {"role": "assistant", "content": "{\"name\": \"get_recommended_books\", \"arguments\": {\"interests\": [\"history\", \"science fiction\"]}}"}, {"role": "observation", "content": "{\"books\": [\"Sapiens: A Brief History of Humankind by Yuval Noah Harari\", \"A Brief History of Time by Stephen Hawking\", \"Dune by Frank Herbert\", \"The Martian by Andy Weir\"]}"}, {"role": "assistant", "content": "Based on your interests in history and science fiction, I would recommend the following books: \"Sapiens: A Brief History of Humankind\" by Yuval Noah Harari, \"A Brief History of Time\" by Stephen Hawking, \"Dune\" by Frank Herbert, and \"The Martian\" by Andy Weir."}]}
```

- `system` 角色为可选角色，但若存在 `system` 角色，其必须出现在 `user`
  角色之前，且一个完整的对话数据（无论单轮或者多轮对话）只能出现一次 `system` 角色。
- `tools` 字段为可选字段，若存在 `tools` 字段，其必须出现在 `system`
  角色之后，且一个完整的对话数据（无论单轮或者多轮对话）只能出现一次 `tools` 字段。当 `tools` 字段存在时，`system`
  角色必须存在并且 `content` 字段为空。

## 配置文件

微调配置文件位于 `config` 目录下，包括以下文件：

1. `ds_zereo_2 / ds_zereo_3.json`: deepspeed 配置文件。
2. `lora.yaml / ptuning_v2
3. .yaml / sft.yaml`: 模型不同方式的配置文件，包括模型参数、优化器参数、训练参数等。 部分重要参数解释如下：
    + data_config 部分
        + train_file: 训练数据集的文件路径。
        + val_file: 验证数据集的文件路径。
        + test_file: 测试数据集的文件路径。
        + num_proc: 在加载数据时使用的进程数量。
    + max_input_length: 输入序列的最大长度。
    + max_output_length: 输出序列的最大长度。
    + training_args 部分
        + output_dir: 用于保存模型和其他输出的目录。
        + max_steps: 训练的最大步数。
        + per_device_train_batch_size: 每个设备（如 GPU）的训练批次大小。
        + dataloader_num_workers: 加载数据时使用的工作线程数量。
        + remove_unused_columns: 是否移除数据中未使用的列。
        + save_strategy: 模型保存策略（例如，每隔多少步保存一次）。
        + save_steps: 每隔多少步保存一次模型。
        + log_level: 日志级别（如 info）。
        + logging_strategy: 日志记录策略。
        + logging_steps: 每隔多少步记录一次日志。
        + per_device_eval_batch_size: 每个设备的评估批次大小。
        + evaluation_strategy: 评估策略（例如，每隔多少步进行一次评估）。
        + eval_steps: 每隔多少步进行一次评估。
        + predict_with_generate: 是否使用生成模式进行预测。
    + generation_config 部分
        + max_new_tokens: 生成的最大新 token 数量。
    + peft_config 部分
        + peft_type: 使用的参数有效调整类型 (支持 LORA 和 PREFIX_TUNING)。
        + task_type: 任务类型，这里是因果语言模型 (不要改动)。
    + Lora 参数：
        + r: LoRA 的秩。
        + lora_alpha: LoRA 的缩放因子。
        + lora_dropout: 在 LoRA 层使用的 dropout 概率。
    + P-TuningV2 参数：
        + num_virtual_tokens: 虚拟 token 的数量。
        + num_attention_heads: 2: P-TuningV2 的注意力头数(不要改动)。
        + token_dim: 256: P-TuningV2 的 token 维度(不要改动)。

## 开始微调

通过以下代码执行 **单机多卡/多机多卡** 运行，这是使用 `deepspeed` 作为加速方案的，您需要安装 `deepspeed`。

```shell
OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc_per_node=8  finetune_hf.py  data/AdvertiseGen/  THUDM/glm-4-9b  configs/lora.yaml
```

通过以下代码执行 **单机单卡** 运行。

```shell
python finetune.py  data/AdvertiseGen/  THUDM/glm-4-9b-chat  configs/lora.yaml
```

## 从保存点进行微调

如果按照上述方式进行训练，每次微调都会从头开始，如果你想从训练一半的模型开始微调，你可以加入第四个参数，这个参数有两种传入方式:

1. `yes`, 自动从最后一个保存的 Checkpoint开始训练
2. `XX`, 断点号数字 例 `600` 则从序号600 Checkpoint开始训练

例如，这就是一个从最后一个保存点继续微调的示例代码

```shell
python finetune.py  data/AdvertiseGen/  THUDM/glm-4-9b-chat  configs/lora.yaml yes
```

## 使用微调后的模型

### 在 inference.py 中验证微调后的模型

您可以在 `finetune_demo/inference.py` 中使用我们的微调后的模型，仅需要一行代码就能简单的进行测试。

```shell
python inference.py your_finetune_path
```

这样，得到的回答就微调后的回答了。

### 在本仓库的其他 demo 或者外部仓库使用微调后的模型

您可以在任何一个 demo 内使用我们的 `LORA` 和 全参微调的模型。这需要你自己按照以下教程进行修改代码。

1. 使用`finetune_demo/inference.py`中读入模型的方式替换 demo 中读入模型的方式。

> 请注意，对于 LORA 和 P-TuningV2 我们没有合并训练后的模型，而是在`adapter_config.json`
> 中记录了微调型的路径，如果你的原始模型位置发生更改，则你应该修改`adapter_config.json`中`base_model_name_or_path`的路径。

```python
def load_model_and_tokenizer(
        model_dir: Union[str, Path], trust_remote_code: bool = True
) -> tuple[ModelType, TokenizerType]:
    model_dir = _resolve_path(model_dir)
    if (model_dir / 'adapter_config.json').exists():
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=trust_remote_code, device_map='auto'
        )
        tokenizer_dir = model.peft_config['default'].base_model_name_or_path
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=trust_remote_code, device_map='auto'
        )
        tokenizer_dir = model_dir
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir, trust_remote_code=trust_remote_code
    )
    return model, tokenizer
```

2. 读取微调的模型，请注意，你应该使用微调模型的位置，例如，若你的模型位置为`/path/to/finetune_adapter_model`
   ，原始模型地址为`path/to/base_model`,则你应该使用`/path/to/finetune_adapter_model`作为`model_dir`。
3. 完成上述操作后，就能正常使用微调的模型了，其他的调用方式没有变化。
4. 本微调脚本没有测试过128K 1M等长文本的微调，长文本的微调需要更大显存的GPU设备，并且需要更高效的微调方案,需要开发者自行解决。

## 参考文献

```

@inproceedings{liu2022p,
title={P-tuning: Prompt tuning can be comparable to fine-tuning across scales and tasks},
author={Liu, Xiao and Ji, Kaixuan and Fu, Yicheng and Tam, Weng and Du, Zhengxiao and Yang, Zhilin and Tang, Jie},
booktitle={Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short
Papers)},
pages={61--68},
year={2022}
}

@misc{tang2023toolalpaca,
title={ToolAlpaca: Generalized Tool Learning for Language Models with 3000 Simulated Cases},
author={Qiaoyu Tang and Ziliang Deng and Hongyu Lin and Xianpei Han and Qiao Liang and Le Sun},
year={2023},
eprint={2306.05301},
archivePrefix={arXiv},
primaryClass={cs.CL}
}

```

# GLM-4-9B Chat dialogue model fine-tuning

In this demo, you will experience how to fine-tune the GLM-4-9B-Chat open source model (visual understanding model is
not supported). Please strictly follow the steps in the document to avoid unnecessary errors.

## Hardware check

**The data in this document are tested in the following hardware environment. The actual operating environment
requirements and the GPU memory occupied by the operation are slightly different. Please refer to the actual operating
environment.**
Test hardware information:

+ OS: Ubuntu 22.04
+ Memory: 512GB
+ Python:  Python: 3.10.12 / 3.12.3 (Currently, you need to install nltk from the git source code if you use Python
  3.12.3)
+ CUDA Version: 12.3
+ GPU Driver: 535.104.05
+ GPU: NVIDIA A100-SXM4-80GB * 8

| Fine-tuning solution | GPU memory usage                             | Weight save point size |
|----------------------|----------------------------------------------|------------------------|
| lora (PEFT)          | 21531MiB                                     | 17M                    |
| p-tuning v2 (PEFT)   | 21381MiB                                     | 121M                   |
| SFT (Zero3 method)   | 80935MiB<br/>(Each GPU, 8 GPUs are required) | 20G                    |

Before starting fine-tuning, please install the dependencies in `basic_demo` first. You also need to install the
dependencies in this directory:

> NOTE: Some codes in NLTK 3.8.1 might not yet be compatible with Python 3.12. For adaptation methods in such cases,
> please refer to [issues #38](https://github.com/THUDM/GLM-4/issues/38).

```bash
pip install -r requirements.txt
```

## Multi-round dialogue format

The multi-round dialogue fine-tuning example uses the GLM-4 dialogue format convention, adding different `loss_mask` to
different roles to calculate `loss` for multiple rounds of replies in one calculation.

For data files, the sample uses the following format:

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

This is a sample without tools:

```
{"messages": [{"role": "user", "content": "类型#裤*材质#牛仔布*风格#性感"}, {"role": "assistant", "content": "3x1的这款牛仔裤采用浅白的牛仔面料为裤身材质，其柔然的手感和细腻的质地，在穿着舒适的同时，透露着清纯甜美的个性气质。除此之外，流畅的裤身剪裁将性感的腿部曲线彰显的淋漓尽致，不失为一款随性出街的必备单品。"}]}
```

This is a sample with tools:

```
{"messages": [{"role": "system", "content": "", "tools": [{"type": "function", "function": {"name": "get_recommended_books", "description": "Get recommended books based on user's interests", "parameters": {"type": "object", "properties": {"interests": {"type": "array", "items": {"type": "string"}, "description": "The interests to recommend books for"}}, "required": ["interests"]}}}]}, {"role": "user", "content": "Hi, I am looking for some book recommendations. I am interested in history and science fiction."}, {"role": "assistant", "content": "{\"name\": \"get_recommended_books\", \"arguments\": {\"interests\": [\"history\", \"science fiction\"]}}"}, {"role": "observation", "content": "{\"books\": [\"Sapiens: A Brief History of Humankind by Yuval Noah Harari\", \"A Brief History of Time by Stephen Hawking\", \"Dune by Frank Herbert\", \"The Martian by Andy Weir\"]}"}, {"role": "assistant", "content": "Based on your interests in history and science fiction, I would recommend the following books: \"Sapiens: A Brief History of Humankind\" by Yuval Noah Harari, \"A Brief History of Time\" by Stephen Hawking, \"Dune\" by Frank Herbert, and \"The Martian\" by Andy Weir."}]}
```

- The `system` role is optional, but if it exists, it must appear before the `user` role, and a complete conversation
  data (whether single-round or multi-round conversation) can only have one `system` role.
- The `tools` field is optional. If it exists, it must appear after the `system` role, and a complete conversation
  data (whether single-round or multi-round conversation) can only have one `tools` field. When the `tools` field
  exists, the `system` role must exist and the `content` field is empty.

## Configuration file

The fine-tuning configuration file is located in the `config` directory, including the following files:

1. `ds_zereo_2 / ds_zereo_3.json`: deepspeed configuration file.

2. `lora.yaml / ptuning_v2
3. .yaml / sft.yaml`: Configuration files for different modes of models, including model parameters, optimizer
   parameters, training parameters, etc. Some important parameters are explained as follows:

+ data_config section
+ train_file: File path of training dataset.
+ val_file: File path of validation dataset.
+ test_file: File path of test dataset.
+ num_proc: Number of processes to use when loading data.
+ max_input_length: Maximum length of input sequence.
+ max_output_length: Maximum length of output sequence.
+ training_args section
+ output_dir: Directory for saving model and other outputs.
+ max_steps: Maximum number of training steps.
+ per_device_train_batch_size: Training batch size per device (such as GPU).
+ dataloader_num_workers: Number of worker threads to use when loading data.
+ remove_unused_columns: Whether to remove unused columns in data.
+ save_strategy: Model saving strategy (for example, how many steps to save).
+ save_steps: How many steps to save the model.
+ log_level: Log level (such as info).
+ logging_strategy: logging strategy.
+ logging_steps: how many steps to log at.
+ per_device_eval_batch_size: per-device evaluation batch size.
+ evaluation_strategy: evaluation strategy (e.g. how many steps to evaluate at).
+ eval_steps: how many steps to evaluate at.
+ predict_with_generate: whether to use generation mode for prediction.
+ generation_config section
+ max_new_tokens: maximum number of new tokens to generate.
+ peft_config section
+ peft_type: type of parameter tuning to use (supports LORA and PREFIX_TUNING).
+ task_type: task type, here is causal language model (don't change).
+ Lora parameters:
+ r: rank of LoRA.
+ lora_alpha: scaling factor of LoRA.
+ lora_dropout: dropout probability to use in LoRA layer.
+ P-TuningV2 parameters:
+ num_virtual_tokens: the number of virtual tokens.
+ num_attention_heads: 2: the number of attention heads of P-TuningV2 (do not change).
+ token_dim: 256: the token dimension of P-TuningV2 (do not change).

## Start fine-tuning

Execute **single machine multi-card/multi-machine multi-card** run through the following code, which uses `deepspeed` as
the acceleration solution, and you need to install `deepspeed`.

```shell
OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc_per_node=8 finetune_hf.py data/AdvertiseGen/ THUDM/glm-4-9b configs/lora.yaml
```

Execute **single machine single card** run through the following code.

```shell
python finetune.py data/AdvertiseGen/ THUDM/glm-4-9b-chat configs/lora.yaml
```

## Fine-tune from a saved point

If you train as described above, each fine-tuning will start from the beginning. If you want to fine-tune from a
half-trained model, you can add a fourth parameter, which can be passed in two ways:

1. `yes`, automatically start training from the last saved Checkpoint

2. `XX`, breakpoint number, for example `600`, start training from Checkpoint 600

For example, this is an example code to continue fine-tuning from the last saved point

```shell
python finetune.py data/AdvertiseGen/ THUDM/glm-4-9b-chat configs/lora.yaml yes
```

## Use the fine-tuned model

### Verify the fine-tuned model in inference.py

You can Use our fine-tuned model in `finetune_demo/inference.py`, and you can easily test it with just one line of code.

```shell
python inference.py your_finetune_path
```

In this way, the answer you get is the fine-tuned answer.

### Use the fine-tuned model in other demos in this repository or external repositories

You can use our `LORA` and fully fine-tuned models in any demo. This requires you to modify the code yourself according
to the following tutorial.

1. Replace the way to read the model in the demo with the way to read the model in `finetune_demo/inference.py`.

> Please note that for LORA and P-TuningV2, we did not merge the trained models, but recorded the fine-tuned path
> in `adapter_config.json`
> If the location of your original model changes, you should modify the path of `base_model_name_or_path`
> in `adapter_config.json`.

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

2. Read the fine-tuned model. Please note that you should use the location of the fine-tuned model. For example, if your
   model location is `/path/to/finetune_adapter_model`
   and the original model address is `path/to/base_model`, you should use `/path/to/finetune_adapter_model`
   as `model_dir`.
3. After completing the above operations, you can use the fine-tuned model normally. Other calling methods remain
   unchanged.
4. This fine-tuning script has not been tested on long texts of 128K or 1M tokens. Fine-tuning long texts requires GPU
   devices with larger memory and more efficient fine-tuning solutions, which developers need to handle on their own.

## Reference

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

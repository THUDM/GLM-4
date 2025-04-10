# GLM-4

<p align="center">
👋 加入我们的 <a href="https://discord.gg/8cnQKdAprg" target="_blank">Discord</a> 和 <a href="resources/WECHAT.md" target="_blank"> 微信 </a>
</p>
<p align="center">
📍在 <a href="https://open.bigmodel.cn/?utm_campaign=open&_channel_track_key=OWTVNma9">智谱AI开放平台</a> 体验和使用更大规模的 GLM 商业模型。
</p>

Read this in [English](README)

## 项目更新

- 🔥🔥 **News**: ```2025/04/14```: 我们发布 `GLM-4-0414` 系列模型，包含 9B 和 32B 两种模型尺寸。
- 🔥 **News**: ``2024/06/18``: 我们发布 [技术报告](https://arxiv.org/pdf/2406.12793), 欢迎查看。
- 🔥 **News**: ``2024/06/05``: 我们发布 `GLM-4-9B` 系列开源模型。

## 模型列表

### GLM-4-0414 系列模型

|        Model        |   Type    | Seq Length |                                                                    Download                                                                    |
|:-------------------:|:---------:|:----------:|:----------------------------------------------------------------------------------------------------------------------------------------------:|
| GLM-4-9B-Chat-0414  |   Chat    |    128K    | [🤗 Huggingface](https://huggingface.co/THUDM/GLM-4-9B-Chat-0414)<br> [🤖 ModelScope](https://modelscope.cn/models/ZhipuAI/GLM-4-9B-Chat-0414) |
|  GLM-4-Z1-9B-0414   | Reasoning |    128K    |   [🤗 Huggingface](https://huggingface.co/THUDM/GLM-4-Z1-9B-0414)<br> [🤖 ModelScope](https://modelscope.cn/models/ZhipuAI/GLM-4-Z1-9B-0414)   |
|   GLM-4-32B-0414    |   Base    |    128K    |   [🤗 Huggingface](https://huggingface.co/THUDM/GLM-4-32B-0414)<br> [🤖 ModelScope](https://modelscope.cn/models/ZhipuAI/GLM-4-9B-Chat-0414)   |
| GLM-4-32B-Chat-0414 |   Chat    |    128K    |  [🤗 Huggingface](https://huggingface.co/THUDM/GLM-4-32B-Chat-0414)<br> [🤖 ModelScope](https://modelscope.cn/models/ZhipuAI/GLM-4-32B-0414)   |
|  GLM-4-Z1-32B-0414  | Reasoning |    128K    |  [🤗 Huggingface](https://huggingface.co/THUDM/GLM-4-Z1-32B-0414)<br> [🤖 ModelScope](https://modelscope.cn/models/ZhipuAI/GLM-4-Z1-32B-0414)  |
|  GLM-4-DR-32B-0414   | Reasoning |    128K    |  [🤗 Huggingface](https://huggingface.co/THUDM/GLM-4-DR-32B-0414)<br> [🤖 ModelScope](https://modelscope.cn/models/ZhipuAI/GLM-4-DR-32B-0414)  |

### GLM-4-9B 系列模型

|        Model        | Type | Seq Length |                                                                                                      Download                                                                                                      |
|:-------------------:|:----:|:----------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|      GLM-4-9B       | Base |     8K     |            [🤗 Huggingface](https://huggingface.co/THUDM/glm-4-9b)<br> [🤖 ModelScope](https://modelscope.cn/models/ZhipuAI/glm-4-9b)<br> [🟣 WiseModel](https://wisemodel.cn/models/ZhipuAI/glm-4-9b)             |
|    GLM-4-9B-Chat    | Chat |    128K    |     [🤗 Huggingface](https://huggingface.co/THUDM/glm-4-9b-chat)<br> [🤖 ModelScope](https://modelscope.cn/models/ZhipuAI/glm-4-9b-chat)<br> [🟣 WiseModel](https://wisemodel.cn/models/ZhipuAI/GLM-4-9B-Chat)     |
|  GLM-4-9B-Chat-HF   | Chat |    128K    |                                     [🤗 Huggingface](https://huggingface.co/THUDM/glm-4-9b-chat-hf)<br> [🤖 ModelScope](https://modelscope.cn/models/ZhipuAI/glm-4-9b-chat-hf)                                     |
|  GLM-4-9B-Chat-1M   | Chat |     1M     |[🤗 Huggingface](https://huggingface.co/THUDM/glm-4-9b-chat-1m)<br> [🤖 ModelScope](https://modelscope.cn/models/ZhipuAI/glm-4-9b-chat-1m)<br> [🟣 WiseModel](https://wisemodel.cn/models/ZhipuAI/GLM-4-9B-Chat-1M) |
| GLM-4-9B-Chat-1M-HF | Chat |     1M     |                                  [🤗 Huggingface](https://huggingface.co/THUDM/glm-4-9b-chat-1m-hf)<br> [🤖 ModelScope](https://modelscope.cn/models/ZhipuAI/glm-4-9b-chat-1m-hf)                                  |
|      GLM-4V-9B      | Chat |     8K     |           [🤗 Huggingface](https://huggingface.co/THUDM/glm-4v-9b)<br> [🤖 ModelScope](https://modelscope.cn/models/ZhipuAI/glm-4v-9b)<br> [🟣 WiseModel](https://wisemodel.cn/models/ZhipuAI/GLM-4V-9B)           |

## 评测结果

### GLM-4-9B 系列

### 对话模型典型任务

| Model               | AlignBench | MT-Bench | IFEval | MMLU | C-Eval | GSM8K | MATH | HumanEval | NaturalCodeBench |
|:--------------------|:----------:|:--------:|:------:|:----:|:------:|:-----:|:----:|:---------:|:----------------:|
| Llama-3-8B-Instruct |    6.40    |   8.00   |  68.6  | 68.4 |  51.3  | 79.6  | 30.0 |   62.2    |       24.7       |
| ChatGLM3-6B         |    5.18    |   5.50   |  28.1  | 61.4 |  69.0  | 72.3  | 25.7 |   58.5    |       11.3       |
| GLM-4-9B-Chat       |    7.01    |   8.35   |  69.0  | 72.4 |  75.6  | 79.6  | 50.6 |   71.8    |       32.2       |

### 基座模型典型任务

| Model               | MMLU | C-Eval | GPQA | GSM8K | MATH | HumanEval |
|:--------------------|:----:|:------:|:----:|:-----:|:----:|:---------:|
| Llama-3-8B          | 66.6 |  51.2  |  -   | 45.8  |  -   |   33.5    |
| Llama-3-8B-Instruct | 68.4 |  51.3  | 34.2 | 79.6  | 30.0 |   62.2    |
| ChatGLM3-6B-Base    | 61.4 |  69.0  | 26.8 | 72.3  | 25.7 |   58.5    |
| GLM-4-9B            | 74.7 |  77.1  | 34.3 | 84.0  | 30.4 |   70.1    |

> 由于 `GLM-4-9B` 在预训练过程中加入了部分数学、推理、代码相关的 instruction 数据，所以将 Llama-3-8B-Instruct 也列入比较范围。

### 长文本

在 1M 的上下文长度下进行[大海捞针实验](https://github.com/LargeWorldModel/LWM/blob/main/scripts/eval_needle.py)，结果如下：

![needle](resources/eval_needle.jpeg)

在 LongBench-Chat 上对长文本能力进行了进一步评测，结果如下:

<p align="center">
<img src="resources/longbench.png" alt="描述文字" style="display: block; margin: auto; width: 65%;">
</p>

### 多语言能力

在六个多语言数据集上对 GLM-4-9B-Chat 和 Llama-3-8B-Instruct 进行了测试，测试结果及数据集对应选取语言如下表

| Dataset     | Llama-3-8B-Instruct | GLM-4-9B-Chat |                                           Languages                                            |
|:------------|:-------------------:|:-------------:|:----------------------------------------------------------------------------------------------:|
| M-MMLU      |        49.6         |     56.6      |                                              all                                               |
| FLORES      |        25.0         |     28.8      | ru, es, de, fr, it, pt, pl, ja, nl, ar, tr, cs, vi, fa, hu, el, ro, sv, uk, fi, ko, da, bg, no |
| MGSM        |        54.0         |     65.3      |                           zh, en, bn, de, es, fr, ja, ru, sw, te, th                           |
| XWinograd   |        61.7         |     73.1      |                                     zh, en, fr, jp, ru, pt                                     |
| XStoryCloze |        84.7         |     90.7      |                           zh, en, ar, es, eu, hi, id, my, ru, sw, te                           |
| XCOPA       |        73.3         |     80.1      |                           zh, et, ht, id, it, qu, sw, ta, th, tr, vi                           |

### 工具调用能力

我们在 [Berkeley Function Calling Leaderboard](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) 上进行了测试并得到了以下结果：

| Model                  | Overall Acc. | AST Summary | Exec Summary | Relevance |
|:-----------------------|:------------:|:-----------:|:------------:|:---------:|
| Llama-3-8B-Instruct    |    58.88     |    59.25    |    70.01     |   45.83   |
| gpt-4-turbo-2024-04-09 |    81.24     |    82.14    |    78.61     |   88.75   |
| ChatGLM3-6B            |    57.88     |    62.18    |    69.78     |   5.42    |
| GLM-4-9B-Chat          |    81.00     |    80.26    |    84.40     |   87.92   |

### 多模态能力

GLM-4V-9B 是一个多模态语言模型，具备视觉理解能力，其相关经典任务的评测结果如下：

|                            | **MMBench-EN-Test** | **MMBench-CN-Test** | **SEEDBench_IMG** | **MMStar** | **MMMU** | **MME** | **HallusionBench** | **AI2D** | **OCRBench** |
|----------------------------|---------------------|---------------------|-------------------|------------|----------|---------|--------------------|----------|--------------|
| **gpt-4o-2024-05-13**      | 83.4                | 82.1                | 77.1              | 63.9       | 69.2     | 2310.3  | 55.0               | 84.6     | 736          |
| **gpt-4-turbo-2024-04-09** | 81.0                | 80.2                | 73.0              | 56.0       | 61.7     | 2070.2  | 43.9               | 78.6     | 656          |
| **gpt-4-1106-preview**     | 77.0                | 74.4                | 72.3              | 49.7       | 53.8     | 1771.5  | 46.5               | 75.9     | 516          |
| **InternVL-Chat-V1.5**     | 82.3                | 80.7                | 75.2              | 57.1       | 46.8     | 2189.6  | 47.4               | 80.6     | 720          |
| **LLaVA-Next-Yi-34B**      | 81.1                | 79.0                | 75.7              | 51.6       | 48.8     | 2050.2  | 34.8               | 78.9     | 574          |
| **Step-1V**                | 80.7                | 79.9                | 70.3              | 50.0       | 49.9     | 2206.4  | 48.4               | 79.2     | 625          |
| **MiniCPM-Llama3-V2.5**    | 77.6                | 73.8                | 72.3              | 51.8       | 45.8     | 2024.6  | 42.4               | 78.4     | 725          |
| **Qwen-VL-Max**            | 77.6                | 75.7                | 72.7              | 49.5       | 52.0     | 2281.7  | 41.2               | 75.7     | 684          |
| **Gemini 1.0 Pro**         | 73.6                | 74.3                | 70.7              | 38.6       | 49.0     | 2148.9  | 45.7               | 72.9     | 680          |
| **Claude 3 Opus**          | 63.3                | 59.2                | 64.0              | 45.7       | 54.9     | 1586.8  | 37.8               | 70.6     | 694          |
| **GLM-4V-9B**              | 81.1                | 79.4                | 76.8              | 58.7       | 47.2     | 2163.8  | 46.6               | 81.1     | 786          |

## 完整项目列表

本开源仓库通过以下内容为开发者提供基础的 GLM-4 开源模型使用和开发代码

+ [inference](inference/README_zh.md)
    + 使用 transformers 和 vLLM 后端的交互代码
    + OpenAI API 后端交互代码
    + Batch 推理代码

+ [finetune](finetune/README_zh)
    + PEFT (LORA, P-Tuning) 微调代码 (适用于 9B模型)
    + SFT 微调代码 (适用于 9B模型)

+ [demo](demo)
    + [intel_device_demo](demo/intel_device_demo) 使用 OpenVINO / 使用 Intel® Extension for Transformers 部署模型代码
    + [composite_demo](demo/composite_demo/README.md) GLM-4-9B-Chat 以及 GLM-4V-9B 开源模型的完整功能演示代码，包含了 All Tools 能力、长文档解读和多模态能力的展示。


## 模型实现代码

如果你想查看我们的模型实现，欢迎查看在相关仓库的模型实现Pull Request，他们已经被合并。

+ [vLLM 模型实现](https://github.com/vllm-project/vllm/pull/16338)
+ [transformers 模型实现](ttps://github.com/huggingface/transformers/pull/37388)
+ [llama.cpp 模型实现]()

## 友情链接

+ [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory): 高效开源微调框架，已支持 GLM-4-9B-Chat 语言模型微调。
+ [SWIFT](https://github.com/modelscope/swift): 魔搭社区的大模型/多模态大模型训练框架，已支持 GLM-4-9B-Chat / GLM-4V-9B 模型微调。
+ [Xorbits Inference](https://github.com/xorbitsai/inference): 性能强大且功能全面的分布式推理框架，轻松一键部署你自己的模型或内置的前沿开源模型。
+ [LangChain-ChatChat](https://github.com/chatchat-space/Langchain-Chatchat): 基于 Langchain 与 ChatGLM 等语言模型的 RAG 与 Agent 应用
+ [self-llm](https://github.com/datawhalechina/self-llm/tree/master/models/GLM-4): Datawhale 团队的提供的 GLM-4-9B 系列模型使用教程。
+ [chatglm.cpp](https://github.com/li-plus/chatglm.cpp): 类似 llama.cpp 的量化加速推理方案，实现笔记本上实时对话
+ [OpenVINO](https://github.com/openvinotoolkit): Intel 开发的高性能 CPU,GPU及NPU 加速推理方案，可以参考此 [步骤](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/llm-chatbot/llm-chatbot-generate-api.ipynb) 部署 glm-4-9b-chat 模型。


## 引用

如果你觉得我们的工作有帮助的话，请考虑引用下列论文。

```
@misc{glm2024chatglm,
      title={ChatGLM: A Family of Large Language Models from GLM-130B to GLM-4 All Tools},
      author={Team GLM and Aohan Zeng and Bin Xu and Bowen Wang and Chenhui Zhang and Da Yin and Diego Rojas and Guanyu Feng and Hanlin Zhao and Hanyu Lai and Hao Yu and Hongning Wang and Jiadai Sun and Jiajie Zhang and Jiale Cheng and Jiayi Gui and Jie Tang and Jing Zhang and Juanzi Li and Lei Zhao and Lindong Wu and Lucen Zhong and Mingdao Liu and Minlie Huang and Peng Zhang and Qinkai Zheng and Rui Lu and Shuaiqi Duan and Shudan Zhang and Shulin Cao and Shuxun Yang and Weng Lam Tam and Wenyi Zhao and Xiao Liu and Xiao Xia and Xiaohan Zhang and Xiaotao Gu and Xin Lv and Xinghan Liu and Xinyi Liu and Xinyue Yang and Xixuan Song and Xunkai Zhang and Yifan An and Yifan Xu and Yilin Niu and Yuantao Yang and Yueyan Li and Yushi Bai and Yuxiao Dong and Zehan Qi and Zhaoyu Wang and Zhen Yang and Zhengxiao Du and Zhenyu Hou and Zihan Wang},
      year={2024},
      eprint={2406.12793},
      archivePrefix={arXiv},
      primaryClass={id='cs.CL' full_name='Computation and Language' is_active=True alt_name='cmp-lg' in_archive='cs' is_general=False description='Covers natural language processing. Roughly includes material in ACM Subject Class I.2.7. Note that work on artificial languages (programming languages, logics, formal systems) that does not explicitly address natural-language issues broadly construed (natural-language processing, computational linguistics, speech, text retrieval, etc.) is not appropriate for this area.'}
}
```

```
@misc{wang2023cogvlm,
      title={CogVLM: Visual Expert for Pretrained Language Models},
      author={Weihan Wang and Qingsong Lv and Wenmeng Yu and Wenyi Hong and Ji Qi and Yan Wang and Junhui Ji and Zhuoyi Yang and Lei Zhao and Xixuan Song and Jiazheng Xu and Bin Xu and Juanzi Li and Yuxiao Dong and Ming Ding and Jie Tang},
      year={2023},
      eprint={2311.03079},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

# GLM-4

<p align="center">
 👋 Join <a href="https://discord.gg/8cnQKdAprg" target="_blank">Discord</a> and <a href="resources/WECHAT.md" target="_blank">WeChat</a>
</p>
<p align="center">
📍Experience and use a larger-scale GLM business model on the <a href="https://open.bigmodel.cn/?utm_campaign=open&_channel_track_key=OWTVNma9">Zhipu AI Open Platform</a>
</p>

[中文阅读](README_zh.md)

## Project Updates

- 🔥🔥 **News**: ```2025/04/14```: We released the [GLM-4-0414](https://huggingface.co/collections/THUDM/glm-4-0414-67f3cbcb34dd9d252707cb2e) model series, available in both 9B and 32B parameter sizes.
- 🔥 **News**: ``2024/06/18``: We published our [technical report](https://arxiv.org/pdf/2406.12793), feel free to check it out.
- 🔥 **News**: ``2024/06/05``: We open-sourced the `GLM-4-9B` model series.

## Performance Showcase

### Animation Rendering

<table>
  <tr>
    <td style="text-align: center; font-size: 16px; font-weight: bold; padding: 10px;">
      GLM-4-Z1-32B-0414
    </td>
    <td style="text-align: center; font-size: 16px; font-weight: bold; padding: 10px;">
      GLM-4-32B-Chat-0414
    </td>
  </tr>
  <tr>
    <td style="vertical-align: top; padding: 10px;">
      <video src="https://github.com/user-attachments/assets/849ff9fd-b54d-4c74-9ee5-3412e1a09e32" 
             style="width: 400px; height: 300px;" autoplay loop muted playsinline></video>
      <div style="margin-top: 10px; font-size: 14px; color: #333; width: 400px;">
        write a Python program that shows a ball bouncing inside a spinning hexagon. The ball should be affected by gravity and friction, and it must bounce off the rotating walls realistically
      </div>
    </td>
    <td style="vertical-align: top; padding: 10px;">
      <video src="https://github.com/user-attachments/assets/ff87767e-66ac-4fcb-935a-eda783d12210" 
             style="width: 400px; height: 300px;" autoplay loop muted playsinline></video>
      <div style="margin-top: 10px; font-size: 14px; color: #333; width: 400px;">
         Create a first person filght simulator with threejs in HTML
      </div>
    </td>
  </tr>
</table>

### Web Design

<table>
  <tr>
    <td style="text-align: center; font-size: 16px; font-weight: bold; padding: 10px;">
      GLM-4-32B-Chat-0414
    </td>
    <td style="text-align: center; font-size: 16px; font-weight: bold; padding: 10px;">
      GLM-4-32B-Chat-0414
    </td>
  </tr>
  <tr>
    <td style="vertical-align: top; padding: 10px;">
      <img src="https://github.com/user-attachments/assets/7474e6af-6874-48c8-83cb-f9b9ea152ba6" style="width: 400px;" />
      <div style="margin-top: 10px; font-size: 14px; color: #333; width: 400px;">
          设计一个支持自定义函数绘制的绘图板，可以添加和删除自定义函数，并为函数指定颜色
      </div>
    </td>
    <td style="vertical-align: top; padding: 10px;">
      <img src="https://github.com/user-attachments/assets/f466ab72-0670-4bde-b007-9ee3f1042480" style="width: 400px;" />
      <div style="margin-top: 10px; font-size: 14px; color: #333; width: 400px;"> 给我设计一个移动端机器学习平台的 UI，其中要包括训练任务，存储管理，和个人统计信息界面。个人信息统计界面要用图表展示用户过去一段时间的各类资源使用情况。使用 Tailwind CSS 来美化页面，把这 3 个手机界面平铺展示到一个 HTML 页面中 </div>
    </td>
  </tr>
</table>


### SVG 

<table>
  <tr>
    <td style="text-align: center; font-size: 16px; font-weight: bold; padding: 10px;">
      GLM-4-32B-Chat-0414
    </td>
    <td style="text-align: center; font-size: 16px; font-weight: bold; padding: 10px;">
      GLM-4-32B-Chat-0414
    </td>
  </tr>
  <tr>
    <td style="vertical-align: top; padding: 10px;">
      <img src="https://github.com/user-attachments/assets/dfecd0d9-dc5b-41c5-9d39-e382976807be" style="width: 400px;" />
      <div style="margin-top: 10px; font-size: 14px; color: #333; width: 400px;">
          用黑白手绘风格给我展示机器学习的本质，画在一张 svg 图中
      </div>
    </td>
    <td style="vertical-align: top; padding: 10px;">
      <img src="https://github.com/user-attachments/assets/986a5d56-23d3-4724-bc70-230088e0d327" style="width: 400px;" />
      <div style="margin-top: 10px; font-size: 14px; color: #333; width: 400px;"> create an svg to illustrate the training pipeline of an LLM </div>
    </td>
  </tr>
</table>


## Model List

### GLM-4-0414 Series

|             Model             |   Type    | Seq Length* |                                                                              Download                                                                               |
|:-----------------------------:|:---------:|:----------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|      GLM-4-9B-Chat-0414       |   Chat    |    32K -> 128K    |           [🤗 Huggingface](https://huggingface.co/THUDM/GLM-4-9B-Chat-0414)<br> [🤖 ModelScope](https://modelscope.cn/models/ZhipuAI/GLM-4-9B-Chat-0414)<br> [🧩 Modelers](https://modelers.cn/models/zhipuai/GLM-4-9B-Chat-0414)            |
|        GLM-Z1-9B-0414         | Reasoning |    32K -> 128K    |             [🤗 Huggingface](https://huggingface.co/THUDM/GLM-4-Z1-9B-0414)<br> [🤖 ModelScope](https://modelscope.cn/models/ZhipuAI/GLM-4-Z1-9B-0414)<br> [🧩 Modelers](https://modelers.cn/models/zhipuai/GLM-4-Z1-9B-0414)              |
|        GLM-4-32B-0414         |   Base    |    32K -> 128K    |             [🤗 Huggingface](https://huggingface.co/THUDM/GLM-4-32B-0414)<br> [🤖 ModelScope](https://modelscope.cn/models/ZhipuAI/GLM-4-9B-Chat-0414)<br> [🧩 Modelers](https://modelers.cn/models/zhipuai/GLM-4-32B-0414)              |
|      GLM-4-32B-Chat-0414      |   Chat    |    32K -> 128K    |             [🤗 Huggingface](https://huggingface.co/THUDM/GLM-4-32B-Chat-0414)<br> [🤖 ModelScope](https://modelscope.cn/models/ZhipuAI/GLM-4-32B-0414)<br> [🧩 Modelers](https://modelers.cn/models/zhipuai/GLM-4-32B-Chat-0414)             |
|        GLM-Z1-32B-0414        | Reasoning |    32K -> 128K    |            [🤗 Huggingface](https://huggingface.co/THUDM/GLM-4-Z1-32B-0414)<br> [🤖 ModelScope](https://modelscope.cn/models/ZhipuAI/GLM-4-Z1-32B-0414)<br> [🧩 Modelers](https://modelers.cn/models/zhipuai/GLM-4-Z1-32B-0414)             |
| GLM-4-Z1-Rumination-32B-0414  | Reasoning |    32K -> 128K    | [🤗 Huggingface](https://huggingface.co/THUDM/GLM-4-Z1-Rumination-32B-0414)<br> [🤖 ModelScope](https://modelscope.cn/models/ZhipuAI/GLM-4-Z1-Rumination-32B-0414)<br> [🧩 Modelers](https://modelers.cn/models/zhipuai/GLM-4-Z1-Rumination-32B-0414)  |


\* The model is natively trained with a 32k context length.* For requests where the total input + output length may exceed 32k, we recommend enabling **Yarn** to achieve better extrapolation performance. You can configure Yarn-based extrapolation using the following settings:

For supported frameworks, you could add the following to `config.json` to enable YaRN:

```json
"rope_scaling": {
    "factor": 4.0,
    "original_max_position_embeddings": 32768,
    "type": "yarn"
}
```

### GLM-4-9B Series

|        Model        | Type | Seq Length |                                                                                                       Download                                                                                                       |
|:-------------------:|:----:|:----------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|      GLM-4-9B       | Base |     8K     |             [🤗 Huggingface](https://huggingface.co/THUDM/glm-4-9b)<br> [🤖 ModelScope](https://modelscope.cn/models/ZhipuAI/glm-4-9b)<br> [🟣 WiseModel](https://wisemodel.cn/models/ZhipuAI/glm-4-9b)              |
|    GLM-4-9B-Chat    | Chat |    128K    |      [🤗 Huggingface](https://huggingface.co/THUDM/glm-4-9b-chat)<br> [🤖 ModelScope](https://modelscope.cn/models/ZhipuAI/glm-4-9b-chat)<br> [🟣 WiseModel](https://wisemodel.cn/models/ZhipuAI/GLM-4-9B-Chat)      |
|  GLM-4-9B-Chat-HF   | Chat |    128K    |                                      [🤗 Huggingface](https://huggingface.co/THUDM/glm-4-9b-chat-hf)<br> [🤖 ModelScope](https://modelscope.cn/models/ZhipuAI/glm-4-9b-chat-hf)                                      |
|  GLM-4-9B-Chat-1M   | Chat |     1M     | [🤗 Huggingface](https://huggingface.co/THUDM/glm-4-9b-chat-1m)<br> [🤖 ModelScope](https://modelscope.cn/models/ZhipuAI/glm-4-9b-chat-1m)<br> [🟣 WiseModel](https://wisemodel.cn/models/ZhipuAI/GLM-4-9B-Chat-1M)  | 
| GLM-4-9B-Chat-1M-HF | Chat |     1M     |                                   [🤗 Huggingface](https://huggingface.co/THUDM/glm-4-9b-chat-1m-hf)<br> [🤖 ModelScope](https://modelscope.cn/models/ZhipuAI/glm-4-9b-chat-1m-hf)                                   |
|      GLM-4V-9B      | Chat |     8K     |            [🤗 Huggingface](https://huggingface.co/THUDM/glm-4v-9b)<br> [🤖 ModelScope](https://modelscope.cn/models/ZhipuAI/glm-4v-9b)<br> [🟣 WiseModel](https://wisemodel.cn/models/ZhipuAI/GLM-4V-9B)            |

## BenchMark

### GLM-4-0414 Series

| Models           | IFEval  | SWE-Bench        | BFCL-v3 (Overall) | BFCL-v3 (MultiTurn) | TAU-Bench (Retail) | TAU-Bench (Airline) | SimpleQA | HotpotQA |
|------------------|---------|------------------|-------------------|---------------------|--------------------|---------------------|----------|----------|
| Qwen2.5-Max      | 85.6    | 24.4             | 50.9              | 30.5                | 58.3               | 22.0                | 79.0     | 52.8     |
| GPT-4o-1120      | 81.9    | 38.8             | 69.6              | 41.0                | 62.8               | 46.0                | 82.8     | 63.9     |
| DeepSeek-V3-0324 | 83.4    | 38.8（oh）         | 66.2              | 35.8                | 60.7               | 32.4                | 82.6     | 54.6     |
| DeepSeek-R1      | 84.3    | 34（oh）/ 49.2（al） | 57.5              | 12.4                | 33.0               | 37.3                | 83.9     | 63.1     |
| GLM-4-32B-0414   | 86.5    |                  | 69.6              | 41.5                | 68.7               | 51.2                | 88.1     | 63.8     |


### GLM-4-9B Series

#### Typical Tasks

| Model               | AlignBench | MT-Bench | IFEval | MMLU | C-Eval | GSM8K | MATH | HumanEval | NaturalCodeBench |
|:--------------------|:----------:|:--------:|:------:|:----:|:------:|:-----:|:----:|:---------:|:----------------:|
| Llama-3-8B-Instruct |    6.40    |   8.00   | 68.58  | 68.4 |  51.3  | 79.6  | 30.0 |   62.2    |       24.7       |
| ChatGLM3-6B         |    5.18    |   5.50   |  28.1  | 66.4 |  69.0  | 72.3  | 25.7 |   58.5    |       11.3       |
| GLM-4-9B-Chat       |    7.01    |   8.35   |  69.0  | 72.4 |  75.6  | 79.6  | 50.6 |   71.8    |       32.2       |

#### Base Model

| Model               | MMLU | C-Eval | GPQA | GSM8K | MATH | HumanEval |
|:--------------------|:----:|:------:|:----:|:-----:|:----:|:---------:|
| Llama-3-8B          | 66.6 |  51.2  |  -   | 45.8  |  -   |   33.5    |
| Llama-3-8B-Instruct | 68.4 |  51.3  | 34.2 | 79.6  | 30.0 |   62.2    |
| ChatGLM3-6B-Base    | 61.4 |  69.0  | 26.8 | 72.3  | 25.7 |   58.5    |
| GLM-4-9B            | 74.7 |  77.1  | 34.3 | 84.0  | 30.4 |   70.1    |

> Since `GLM-4-9B` adds some math, reasoning, and code-related instruction data during pre-training, Llama-3-8B-Instruct
> is also included in the comparison range.

#### Long Context

The [needle-in-the-haystack experiment](https://github.com/LargeWorldModel/LWM/blob/main/scripts/eval_needle.py) was
conducted with a context length of 1M, and the results are as follows:

![needle](resources/eval_needle.jpeg)

The long text capability was further evaluated on LongBench-Chat, and the results are as follows:

<p align="center">
<img src="resources/longbench.png" alt="Description text" style="display: block; margin: auto; width: 65%;">
</p>

#### Multi Language

The tests for GLM-4-9B-Chat and Llama-3-8B-Instruct are conducted on six multilingual datasets. The test results and the corresponding languages selected for each dataset are shown in the table below:

| Dataset     | Llama-3-8B-Instruct | GLM-4-9B-Chat |                                           Languages                                            |
|:------------|:-------------------:|:-------------:|:----------------------------------------------------------------------------------------------:|
| M-MMLU      |        49.6         |     56.6      |                                              all                                               |
| FLORES      |        25.0         |     28.8      | ru, es, de, fr, it, pt, pl, ja, nl, ar, tr, cs, vi, fa, hu, el, ro, sv, uk, fi, ko, da, bg, no |
| MGSM        |        54.0         |     65.3      |                           zh, en, bn, de, es, fr, ja, ru, sw, te, th                           |
| XWinograd   |        61.7         |     73.1      |                                     zh, en, fr, jp, ru, pt                                     |
| XStoryCloze |        84.7         |     90.7      |                           zh, en, ar, es, eu, hi, id, my, ru, sw, te                           |
| XCOPA       |        73.3         |     80.1      |                           zh, et, ht, id, it, qu, sw, ta, th, tr, vi                           |

#### Function Call

Tested on [Berkeley Function Calling Leaderboard](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard).

| Model                  | Overall Acc. | AST Summary | Exec Summary | Relevance |
|:-----------------------|:------------:|:-----------:|:------------:|:---------:|
| Llama-3-8B-Instruct    |    58.88     |    59.25    |    70.01     |   45.83   |
| gpt-4-turbo-2024-04-09 |    81.24     |    82.14    |    78.61     |   88.75   |
| ChatGLM3-6B            |    57.88     |    62.18    |    69.78     |   5.42    |
| GLM-4-9B-Chat          |    81.00     |    80.26    |    84.40     |   87.92   |

#### Multi-Modal

GLM-4V-9B is a multimodal language model with visual understanding capabilities. The evaluation results of its related classic tasks are as follows:

|                            | **MMBench-EN-Test** | **MMBench-CN-Test** | **SEEDBench_IMG** | **MMStar** | **MMMU** | **MME** | **HallusionBench** | **AI2D** | **OCRBench** |
|----------------------------|---------------------|---------------------|-------------------|------------|----------|---------|--------------------|----------|--------------|
| **gpt-4o-2024-05-13**      | 83.4                | 82.1                | 77.1              | 63.9       | 69.2     | 2310.3  | 55                 | 84.6     | 736          |
| **gpt-4-turbo-2024-04-09** | 81.0                | 80.2                | 73.0              | 56.0       | 61.7     | 2070.2  | 43.9               | 78.6     | 656          |
| **gpt-4-1106-preview**     | 77.0                | 74.4                | 72.3              | 49.7       | 53.8     | 1771.5  | 46.5               | 75.9     | 516          |
| **InternVL-Chat-V1.5**     | 82.3                | 80.7                | 75.2              | 57.1       | 46.8     | 2189.6  | 47.4               | 80.6     | 720          |
| **LLaVA-Next-Yi-34B**      | 81.1                | 79                  | 75.7              | 51.6       | 48.8     | 2050.2  | 34.8               | 78.9     | 574          |
| **Step-1V**                | 80.7                | 79.9                | 70.3              | 50.0       | 49.9     | 2206.4  | 48.4               | 79.2     | 625          |
| **MiniCPM-Llama3-V2.5**    | 77.6                | 73.8                | 72.3              | 51.8       | 45.8     | 2024.6  | 42.4               | 78.4     | 725          |
| **Qwen-VL-Max**            | 77.6                | 75.7                | 72.7              | 49.5       | 52       | 2281.7  | 41.2               | 75.7     | 684          |
| **Gemini 1.0 Pro**         | 73.6                | 74.3                | 70.7              | 38.6       | 49       | 2148.9  | 45.7               | 72.9     | 680          |
| **Claude 3 Opus**          | 63.3                | 59.2                | 64                | 45.7       | 54.9     | 1586.8  | 37.8               | 70.6     | 694          |
| **GLM-4V-9B**              | 81.1                | 79.4                | 76.8              | 58.7       | 47.2     | 2163.8  | 46.6               | 81.1     | 786          |

## Project list

This repository provides developers with the basic usage and development code for the GLM-4 open-source models through the following components:

+ [inference](inference/README.md)
    + Interactive code using transformers and vLLM backends
    + OpenAI API backend interaction code
    + Batch inference code

+ [finetune](finetune/README.md)
    + PEFT (LoRA, P-Tuning) fine-tuning code (for 9B models)
    + SFT fine-tuning code (for 9B models)

+ [demo](demo)
    + [intel_device_demo](demo/intel_device_demo) Deployment code using OpenVINO / Intel® Extension for Transformers
    + [composite_demo](demo/composite_demo/README.md) Full-feature demo code for GLM-4-9B-Chat and GLM-4V-9B open-source models, including All Tools capability, long-document understanding, and multimodal features. This does **not** apply to the `GLM-4-0414` models.

## Model Implementation Code

If you're interested in our model implementation, feel free to check out the related Pull Requests in the respective repositories — they have already been merged.

+ [vLLM Model Implementation](https://github.com/vllm-project/vllm/pull/16338)  
+ [transformers Model Implementation](https://github.com/huggingface/transformers/pull/37388)  
+ [llama.cpp Model Implementation](https://github.com/ggml-org/llama.cpp/pull/12867)

## Friendly Links

+ [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory): Efficient open-source fine-tuning framework,
  already supports GLM-4-9B-Chat language model fine-tuning.
+ [SWIFT](https://github.com/modelscope/swift): LLM/VLM training framework from ModelScope, supports
  GLM-4-9B-Chat / GLM-4V-9b fine-tuning.
+ [Xorbits Inference](https://github.com/xorbitsai/inference): Performance-enhanced and comprehensive global inference
  framework, easily deploy your own models or import cutting-edge open source models with one click.
+ [LangChain-ChatChat](https://github.com/chatchat-space/Langchain-Chatchat): RAG and Agent applications based on
  language models such as Langchain and ChatGLM
+ [self-llm](https://github.com/datawhalechina/self-llm/tree/master/models/GLM-4): Datawhale's self-llm project, which
  includes
  the GLM-4-9B open source model cookbook.
+ [chatglm.cpp](https://github.com/li-plus/chatglm.cpp): Real-time inference on your laptop accelerated by quantization,
  similar to llama.cpp.
+ [OpenVINO](https://github.com/openvinotoolkit): glm-4-9b-chat already supports the use of OpenVINO. The toolkit accelerates inference and has a greater inference speed improvement on Intel's GPU, GPU and NPU devices. For
specific usage, please refer to  [OpenVINO notebooks](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/llm-chatbot/llm-chatbot-generate-api.ipynb)

## Reference

If you find our work helpful, please consider citing the following paper.

```
@misc{glm2024chatglm,
      title={ChatGLM: A Family of Large Language Models from GLM-130B to GLM-4 All Tools},
      author={Team GLM  and Aohan Zeng and Bin Xu and Bowen Wang and Chenhui Zhang and Da Yin and Diego Rojas and Guanyu Feng and Hanlin Zhao and Hanyu Lai and Hao Yu and Hongning Wang and Jiadai Sun and Jiajie Zhang and Jiale Cheng and Jiayi Gui and Jie Tang and Jing Zhang and Juanzi Li and Lei Zhao and Lindong Wu and Lucen Zhong and Mingdao Liu and Minlie Huang and Peng Zhang and Qinkai Zheng and Rui Lu and Shuaiqi Duan and Shudan Zhang and Shulin Cao and Shuxun Yang and Weng Lam Tam and Wenyi Zhao and Xiao Liu and Xiao Xia and Xiaohan Zhang and Xiaotao Gu and Xin Lv and Xinghan Liu and Xinyi Liu and Xinyue Yang and Xixuan Song and Xunkai Zhang and Yifan An and Yifan Xu and Yilin Niu and Yuantao Yang and Yueyan Li and Yushi Bai and Yuxiao Dong and Zehan Qi and Zhaoyu Wang and Zhen Yang and Zhengxiao Du and Zhenyu Hou and Zihan Wang},
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

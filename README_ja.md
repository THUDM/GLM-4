# GLM-4

<p align="center">
 📄<a href="https://arxiv.org/pdf/2406.12793" target="_blank"> レポート </a> • 🤗 <a href="https://huggingface.co/collections/THUDM/glm-4-665fcf188c414b03c2f7e3b7" target="_blank">HF リポジトリ</a> • 🤖 <a href="https://modelscope.cn/models/ZhipuAI/glm-4-9b-chat" target="_blank">ModelScope</a>  • 🟣 <a href="https://wisemodel.cn/models/ZhipuAI/glm-4-9b-chat" target="_blank">WiseModel</a>  • 🐦 <a href="https://twitter.com/thukeg" target="_blank">Twitter</a> • 👋 <a href="https://discord.gg/8cnQKdAprg" target="_blank">Discord</a> と <a href="resources/WECHAT.md" target="_blank">WeChat</a> に参加
</p>
<p align="center">
📍<a href="https://open.bigmodel.cn/?utm_campaign=open&_channel_track_key=OWTVNma9">Zhipu AI オープンプラットフォーム</a> でより大規模な GLM ビジネスモデルを体験および使用
</p>

[Englidsh](README.md) | [中文](README_zh.md) で読む

## 更新情報

- 🔥🔥 **ニュース**: ```2024/11/01```: 本リポジトリの依存関係が更新されました。モデルが正しく動作するように、`requirements.txt` の依存関係を更新してください。 [glm-4-9b-chat-hf](https://huggingface.co/THUDM/glm-4-9b-chat-hf) のモデルウェイトは `transformers>=4.46.2` と互換性があり、`transformers` ライブラリの `GlmModel` クラスを使用して実装できます。また、 [glm-4-9b-chat](https://huggingface.co/THUDM/glm-4-9b-chat) および [glm-4v-9b](https://huggingface.co/THUDM/glm-4v-9b) の `tokenizer_chatglm.py` が最新バージョンの `transformers` に対応するように更新されました。HuggingFace でファイルを更新してください。
- 🔥 **ニュース**: ```2024/10/27```: [LongReward](https://github.com/THUDM/LongReward) をオープンソース化しました。これは、AI フィードバックを使用して長いコンテキストの大規模言語モデルを強化するモデルです。
- 🔥 **ニュース**: ```2024/10/25```: エンドツーエンドの中国語-英語音声対話モデル [GLM-4-Voice](https://github.com/THUDM/GLM-4-Voice) をオープンソース化しました。
- 🔥 **ニュース**: ```2024/09/05```: 長いコンテキストの Q&A で LLM が細かい引用を生成できるようにするモデル [longcite-glm4-9b](https://huggingface.co/THUDM/LongCite-glm4-9b) とデータセット [LongCite-45k](https://huggingface.co/datasets/THUDM/LongCite-45k) をオープンソース化しました。 [Huggingface Space](https://huggingface.co/spaces/THUDM/LongCite) でオンラインで試してみてください。
- 🔥 **ニュース**: ```2024/08/15```: 単一ターンの対話で 10,000 トークン以上を生成できるモデル [longwriter-glm4-9b](https://huggingface.co/THUDM/LongWriter-glm4-9b) とデータセット [LongWriter-6k](https://huggingface.co/datasets/THUDM/LongWriter-6k) をオープンソース化しました。 [Huggingface Space](https://huggingface.co/spaces/THUDM/LongWriter) または [ModelScope Community Space](https://modelscope.cn/studios/ZhipuAI/LongWriter-glm4-9b-demo) でオンラインで体験してください。
- 🔥 **ニュース**: ```2024/07/24```: 長文処理に関する最新の技術的洞察を公開しました。オープンソース GLM-4-9B モデルの長文トレーニングに関する技術レポートを [こちら](https://medium.com/@ChatGLM/glm-long-scaling-pre-trained-model-contexts-to-millions-caa3c48dea85) でご覧ください。
- 🔥 **ニュース**: ```2024/07/09```: GLM-4-9B-Chat モデルが [Ollama](https://github.com/ollama/ollama) と [Llama.cpp](https://github.com/ggerganov/llama.cpp) に対応しました。詳細は [PR](https://github.com/ggerganov/llama.cpp/pull/8031) をご覧ください。
- 🔥 **ニュース**: ```2024/06/18```: [技術レポート](https://arxiv.org/pdf/2406.12793) を公開しました。ご覧ください。
- 🔥 **ニュース**: ```2024/06/05```: GLM-4-9B シリーズのオープンソースモデルをリリースしました。

## モデル紹介

GLM-4-9B は、Zhipu AI がリリースした最新世代の GLM-4 シリーズのオープンソース版です。セマンティクス、数学、推論、コード、知識のデータセット評価において、**GLM-4-9B** とその人間の好みに合わせたバージョン **GLM-4-9B-Chat** は、Llama-3-8B を超える優れた性能を示しました。マルチラウンドの会話に加えて、GLM-4-9B-Chat は、ウェブブラウジング、コード実行、カスタムツール呼び出し（Function Call）、長文推論（最大 128K コンテキストをサポート）などの高度な機能も備えています。この世代のモデルは多言語サポートを追加し、日本語、韓国語、ドイツ語を含む 26 言語をサポートしています。また、1M コンテキスト長（約 200 万文字）をサポートする **GLM-4-9B-Chat-1M** モデルと、GLM-4-9B に基づくマルチモーダルモデル GLM-4V-9B もリリースしました。**GLM-4V-9B** は、1120*1120 の高解像度での中英二言語の対話能力を持ち、中英総合能力、知覚＆推論、文字認識、チャート理解などの多方面のマルチモーダル評価において、GPT-4-turbo-2024-04-09、Gemini 1.0 Pro、Qwen-VL-Max、Claude 3 Opus を超える優れた性能を示しました。

## モデルリスト

|        モデル        | タイプ | シーケンス長 | Transformers バージョン |                                                                                                      ダウンロード                                                                                                       |                                                                                        オンラインデモ                                                                                         |
|:-------------------:|:----:|:----------:|:--------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|      GLM-4-9B       | ベース |     8K     |  `4.44.0 - 4.45.0`   |             [🤗 Huggingface](https://huggingface.co/THUDM/glm-4-9b)<br> [🤖 ModelScope](https://modelscope.cn/models/ZhipuAI/glm-4-9b)<br> [🟣 WiseModel](https://wisemodel.cn/models/ZhipuAI/glm-4-9b)             |                                                                                             /                                                                                              |
|    GLM-4-9B-Chat    | チャット |    128K    |     `>= 4.44.0`      |     [🤗 Huggingface](https://huggingface.co/THUDM/glm-4-9b-chat)<br> [🤖 ModelScope](https://modelscope.cn/models/ZhipuAI/glm-4-9b-chat)<br> [🟣 WiseModel](https://wisemodel.cn/models/ZhipuAI/GLM-4-9B-Chat)      | [🤖 ModelScope CPU](https://modelscope.cn/studios/dash-infer/GLM-4-Chat-DashInfer-Demo/summary)<br> [🤖 ModelScope vLLM](https://modelscope.cn/studios/ZhipuAI/glm-4-9b-chat-vllm/summary) |
|  GLM-4-9B-Chat-HF   | チャット |    128K    |     `>= 4.46.0`      |                                     [🤗 Huggingface](https://huggingface.co/THUDM/glm-4-9b-chat-hf)<br> [🤖 ModelScope](https://modelscope.cn/models/ZhipuAI/glm-4-9b-chat-hf)                                      | [🤖 ModelScope CPU](https://modelscope.cn/studios/dash-infer/GLM-4-Chat-DashInfer-Demo/summary)<br> [🤖 ModelScope vLLM](https://modelscope.cn/studios/ZhipuAI/glm-4-9b-chat-vllm/summary) |
|  GLM-4-9B-Chat-1M   | チャット |     1M     |     `>= 4.44.0`      | [🤗 Huggingface](https://huggingface.co/THUDM/glm-4-9b-chat-1m)<br> [🤖 ModelScope](https://modelscope.cn/models/ZhipuAI/glm-4-9b-chat-1m)<br> [🟣 WiseModel](https://wisemodel.cn/models/ZhipuAI/GLM-4-9B-Chat-1M) |                                                                                             /                                                                                              |
| GLM-4-9B-Chat-1M-HF | チャット |     1M     |     `>= 4.46.0`      |                                  [🤗 Huggingface](https://huggingface.co/THUDM/glm-4-9b-chat-1m-hf)<br> [🤖 ModelScope](https://modelscope.cn/models/ZhipuAI/glm-4-9b-chat-1m-hf)                                   |                                                                                             /                                                                                              |
|      GLM-4V-9B      | チャット |     8K     |     `>= 4.46.0`      |           [🤗 Huggingface](https://huggingface.co/THUDM/glm-4v-9b)<br> [🤖 ModelScope](https://modelscope.cn/models/ZhipuAI/glm-4v-9b)<br> [🟣 WiseModel](https://wisemodel.cn/models/ZhipuAI/GLM-4V-9B)            |                                                       [🤖 ModelScope](https://modelscope.cn/studios/ZhipuAI/glm-4v-9b-Demo/summary)                                                        |

## ベンチマーク

### 典型的なタスク

| モデル               | AlignBench | MT-Bench | IFEval | MMLU | C-Eval | GSM8K | MATH | HumanEval | NaturalCodeBench |
|:--------------------|:----------:|:--------:|:------:|:----:|:------:|:-----:|:----:|:---------:|:----------------:|
| Llama-3-8B-Instruct |    6.40    |   8.00   | 68.58  | 68.4 |  51.3  | 79.6  | 30.0 |   62.2    |       24.7       |
| ChatGLM3-6B         |    5.18    |   5.50   |  28.1  | 66.4 |  69.0  | 72.3  | 25.7 |   58.5    |       11.3       |
| GLM-4-9B-Chat       |    7.01    |   8.35   |  69.0  | 72.4 |  75.6  | 79.6  | 50.6 |   71.8    |       32.2       |

### ベースモデル

| モデル               | MMLU | C-Eval | GPQA | GSM8K | MATH | HumanEval |
|:--------------------|:----:|:------:|:----:|:-----:|:----:|:---------:|
| Llama-3-8B          | 66.6 |  51.2  |  -   | 45.8  |  -   |   33.5    |
| Llama-3-8B-Instruct | 68.4 |  51.3  | 34.2 | 79.6  | 30.0 |   62.2    |
| ChatGLM3-6B-Base    | 61.4 |  69.0  | 26.8 | 72.3  | 25.7 |   58.5    |
| GLM-4-9B            | 74.7 |  77.1  | 34.3 | 84.0  | 30.4 |   70.1    |

> `GLM-4-9B` は、事前トレーニング中に数学、推論、コードに関連する一部のインストラクションデータを追加しているため、Llama-3-8B-Instruct も比較範囲に含めています。

### 長いコンテキスト

1M のコンテキスト長で [needle-in-the-haystack experiment](https://github.com/LargeWorldModel/LWM/blob/main/scripts/eval_needle.py) を実施し、結果は以下の通りです：

![needle](resources/eval_needle.jpeg)

LongBench-Chat で長文能力をさらに評価し、結果は以下の通りです：

<p align="center">
<img src="resources/longbench.png" alt="説明文" style="display: block; margin: auto; width: 65%;">
</p>

### 多言語

GLM-4-9B-Chat と Llama-3-8B-Instruct を 6 つの多言語データセットでテストしました。テスト結果と各データセットに対応する言語は以下の表の通りです：

| データセット     | Llama-3-8B-Instruct | GLM-4-9B-Chat |                                           言語                                            |
|:------------|:-------------------:|:-------------:|:----------------------------------------------------------------------------------------------:|
| M-MMLU      |        49.6         |     56.6      |                                              all                                               |
| FLORES      |        25.0         |     28.8      | ru, es, de, fr, it, pt, pl, ja, nl, ar, tr, cs, vi, fa, hu, el, ro, sv, uk, fi, ko, da, bg, no |
| MGSM        |        54.0         |     65.3      |                           zh, en, bn, de, es, fr, ja, ru, sw, te, th                           |
| XWinograd   |        61.7         |     73.1      |                                     zh, en, fr, jp, ru, pt                                     |
| XStoryCloze |        84.7         |     90.7      |                           zh, en, ar, es, eu, hi, id, my, ru, sw, te                           |
| XCOPA       |        73.3         |     80.1      |                           zh, et, ht, id, it, qu, sw, ta, th, tr, vi                           |

### 関数呼び出し

[Berkeley Function Calling Leaderboard](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) でテストしました。

| モデル                  | 全体の精度 | AST サマリー | 実行サマリー | 関連性 |
|:-----------------------|:------------:|:-----------:|:------------:|:---------:|
| Llama-3-8B-Instruct    |    58.88     |    59.25    |    70.01     |   45.83   |
| gpt-4-turbo-2024-04-09 |    81.24     |    82.14    |    78.61     |   88.75   |
| ChatGLM3-6B            |    57.88     |    62.18    |    69.78     |   5.42    |
| GLM-4-9B-Chat          |    81.00     |    80.26    |    84.40     |   87.92   |

### マルチモーダル

GLM-4V-9B は視覚理解能力を持つマルチモーダル言語モデルです。関連するクラシックタスクの評価結果は以下の通りです：

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

## クイックコール

**ハードウェア構成とシステム要件については、[こちら](basic_demo/README_en.md) をご覧ください。**

### GLM-4-9B-Chat 言語モデルを迅速に呼び出す方法

transformers バックエンドを使用して推論を行う：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

os.environ[
    'CUDA_VISIBLE_DEVICES'] = '0'  # GPU 番号を設定します。複数の GPU で推論する場合は、複数の GPU 番号を設定します
MODEL_PATH = "THUDM/glm-4-9b-chat-hf"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

query = "こんにちは"

inputs = tokenizer.apply_chat_template([{"role": "user", "content": query}],
                                       add_generation_prompt=True,
                                       tokenize=True,
                                       return_tensors="pt",
                                       return_dict=True
                                       )

inputs = inputs.to(device)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    device_map="auto"
).eval()

gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
with torch.no_grad():
    outputs = model.generate(**inputs, **gen_kwargs)
    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

vLLM バックエンドを使用して推論を行う：

```python
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# GLM-4-9B-Chat
# OOM が発生した場合は、max_model_len を減らすか、tp_size を増やしてみてください
max_model_len, tp_size = 131072, 1
model_name = "THUDM/glm-4-9b-chat-hf"
prompt = [{"role": "user", "content": "こんにちは"}]

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
llm = LLM(
    model=model_name,
    tensor_parallel_size=tp_size,
    max_model_len=max_model_len,
    trust_remote_code=True,
    enforce_eager=True,
    # GLM-4-9B-Chat-1M で OOM が発生した場合は、以下のパラメータを有効にしてみてください
    # enable_chunked_prefill=True,
    # max_num_batched_tokens=8192
)
stop_token_ids = [151329, 151336, 151338]
sampling_params = SamplingParams(temperature=0.95, max_tokens=1024, stop_token_ids=stop_token_ids)

inputs = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
outputs = llm.generate(prompts=inputs, sampling_params=sampling_params)

print(outputs[0].outputs[0].text)

```

### GLM-4V-9B マルチモーダルモデルを迅速に呼び出す方法

transformers バックエンドを使用して推論を行う：

```python
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

os.environ[
    'CUDA_VISIBLE_DEVICES'] = '0'  # GPU 番号を設定します。複数の GPU で推論する場合は、複数の GPU 番号を設定します
MODEL_PATH = "THUDM/glm-4v-9b"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

query = 'この画像の内容を説明してください'
image = Image.open("your image").convert('RGB')
inputs = tokenizer.apply_chat_template([{"role": "user", "image": image, "content": query}],
                                       add_generation_prompt=True, tokenize=True, return_tensors="pt",
                                       return_dict=True)  # チャットモード

inputsをデバイスに転送
inputs = inputs.to(device)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    device_map="auto"
).eval()

gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
with torch.no_grad():
    outputs = model.generate(**inputs, **gen_kwargs)
    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    print(tokenizer.decode(outputs[0]))
```

vLLM バックエンドを使用して推論を行う：

```python
from PIL import Image
from vllm import LLM, SamplingParams

model_name = "THUDM/glm-4v-9b"

llm = LLM(model=model_name,
          tensor_parallel_size=1,
          max_model_len=8192,
          trust_remote_code=True,
          enforce_eager=True)
stop_token_ids = [151329, 151336, 151338]
sampling_params = SamplingParams(temperature=0.2,
                                 max_tokens=1024,
                                 stop_token_ids=stop_token_ids)

prompt = "画像の内容は何ですか？"
image = Image.open("your image").convert('RGB')
inputs = {
    "prompt": prompt,
    "multi_modal_data": {
        "image": image
    },
}
outputs = llm.generate(inputs, sampling_params=sampling_params)

for o in outputs:
    generated_text = o.outputs[0].text
    print(generated_text)

```

## 完全なプロジェクトリスト

GLM-4-9B シリーズのオープンソースモデルについて詳しく知りたい場合、このオープンソースリポジトリは、以下の内容を通じて開発者に基本的な GLM-4-9B の使用および開発コードを提供します

+ [basic_demo](basic_demo/README.md): 含まれる内容
  + transformers および vLLM バックエンドを使用したインタラクションコード
  + OpenAI API バックエンドインタラクションコード
  + バッチ推論コード

+ [composite_demo](composite_demo/README.md): 含まれる内容
  + GLM-4-9B および GLM-4V-9B オープンソースモデルの完全な機能デモコード、All Tools 機能、長文解釈、およびマルチモーダル機能を含む。

+ [fintune_demo](finetune_demo/README.md): 含まれる内容
  + PEFT (LORA, P-Tuning) 微調整コード
  + SFT 微調整コード

+ [intel_device_demo](intel_device_demo/): 含まれる内容
  + OpenVINO 展開コード
  + Intel® Extension for Transformers 展開コード

## 友好的なリンク

+ [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory): 高効率のオープンソース微調整フレームワーク、GLM-4-9B-Chat 言語モデルの微調整をサポートしています。
+ [SWIFT](https://github.com/modelscope/swift): ModelScope の LLM/VLM トレーニングフレームワーク、GLM-4-9B-Chat / GLM-4V-9b の微調整をサポートしています。
+ [Xorbits Inference](https://github.com/xorbitsai/inference): パフォーマンスが向上し、包括的なグローバル推論フレームワーク、独自のモデルを簡単にデプロイするか、最先端のオープンソースモデルをワンクリックでインポートできます。
+ [LangChain-ChatChat](https://github.com/chatchat-space/Langchain-Chatchat): Langchain や ChatGLM などの言語モデルに基づく RAG およびエージェントアプリケーション
+ [self-llm](https://github.com/datawhalechina/self-llm/tree/master/models/GLM-4): Datawhale の self-llm プロジェクト、GLM-4-9B オープンソースモデルのクックブックを含む。
+ [chatglm.cpp](https://github.com/li-plus/chatglm.cpp): 量子化によってラップトップでリアルタイム推論を実現、llama.cpp に似ています。
+ [OpenVINO](https://github.com/openvinotoolkit): glm-4-9b-chat はすでに OpenVINO を使用してサポートしています。このツールキットは推論を加速し、Intel の GPU、GPU、および NPU デバイスでの推論速度の向上を実現します。具体的な使用方法については、 [OpenVINO notebooks](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/llm-chatbot/llm-chatbot-generate-api.ipynb) を参照してください。

## ライセンス

+ GLM-4 モデルウェイトの使用は、 [モデルライセンス](https://huggingface.co/THUDM/glm-4-9b/blob/main/LICENSE) に従う必要があります。

+ このオープンソースリポジトリのコードは [Apache 2.0](LICENSE) ライセンスに従います。

オープンソースライセンスを厳守してください。

## 引用

私たちの仕事が役立つと思われる場合は、以下の論文を引用してください。

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

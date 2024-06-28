# Deploy the GLM-4-9B-Chat model using OpenVINO

[OpenVINO](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html)
is an open source toolkit designed by Intel for deep learning inference. It can help developers optimize models, improve inference performance, and reduce model memory usage.
This example will show how to deploy the GLM-4-9B-Chat model using OpenVINO.

## 1. Environment configuration

First, you need to install the dependencies

```bash
pip install -r requirements.txt
```

## 2. Convert the model

Since the Huggingface model needs to be converted to an OpenVINO IR model, you need to download the model and convert it.

```
python3 convert.py --model_id THUDM/glm-4-9b-chat --output {your_path}/glm-4-9b-chat-ov
```
The conversion process is as follows:
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

### Optional parameters

* `--model_id` - Path to the directory where the model is located (absolute path).

* `--output` - Path to where the converted model is saved.

* `--precision` - Precision of the conversion.

## 3. Run the GLM-4-9B-Chat model

```
python3 chat.py --model_path {your_path}glm-4-9b-chat-ov --max_sequence_length 4096 --device CPU
```

### Optional parameters

* `--model_path` - Path to the directory where the OpenVINO IR model is located.

* `--max_sequence_length` - Maximum size of the output token.
* `--device` - the device to run inference on.

### Reference code

This code is modified based on the [OpenVINO official example](https://github.com/OpenVINO-dev-contest/chatglm3.openvino).
# 使用 Intel® Extension for Transformers 推理 GLM-4-9B-Chat 模型

本示例介绍如何使用 Intel® Extension for Transformers 推理 GLM-4-9B-Chat 模型。

## 设备和依赖检查

### 相关推理测试数据

**本文档的数据均在以下硬件环境测试,实际运行环境需求和运行占用的显存略有不同，请以实际运行环境为准。**

测试硬件信息:

+ OS: Ubuntu 22.04 (本教程一定需要在Linux环境下执行)
+ Memory: 512GB 
+ Python: 3.10.12 
+ CPU: Intel(R) Xeon(R) Platinum 8358 CPU / 12th Gen Intel i5-12400

## 安装依赖

在开始推理之前，请你先安装`basic_demo`中的依赖，同时您需要安装本目录下的依赖项：
```shell
pip install -r requirements.txt
```

## 运行模型推理

```shell
python itrex_cli_demo.py
```

如果您是第一次推理，会有一次模型转换权重的过程，转换后的模型权重存放在`runtime_outputs`文件夹下，这大概会消耗`60G`的硬盘空间。
转换完成后，文件夹下有两个文件：
+ ne_chatglm2_f32.bin 52G(如果您不使用FP32进行推理，可以删掉这个文件)
+ ne_chatglm2_q_nf4_bestla_cfp32_sym_sfp32_g32.bin 8.1G

如果您不是第一次推理，则会跳过这个步骤，直接开始对话，推理效果如下：
```shell
Welcome to the CLI chat. Type your messages below.

User: 你好
AVX:1 AVX2:1 AVX512F:1 AVX512BW:1 AVX_VNNI:0 AVX512_VNNI:1 AMX_INT8:0 AMX_BF16:0 AVX512_BF16:0 AVX512_FP16:0
beam_size: 1, do_sample: 1, top_k: 40, top_p: 0.900, continuous_batching: 0, max_request_num: 1, early_stopping: 0, scratch_size_ratio: 1.000
model_file_loader: loading model from runtime_outs/ne_chatglm2_q_nf4_bestla_cfp32_sym_sfp32_g32.bin
Loading the bin file with NE format...
load_ne_hparams  0.hparams.n_vocab = 151552                        
load_ne_hparams  1.hparams.n_embd = 4096                          
load_ne_hparams  2.hparams.n_mult = 0                             
load_ne_hparams  3.hparams.n_head = 32                            
load_ne_hparams  4.hparams.n_head_kv = 0                             
load_ne_hparams  5.hparams.n_layer = 40                            
load_ne_hparams  6.hparams.n_rot = 0                             
load_ne_hparams  7.hparams.ftype = 0                             
load_ne_hparams  8.hparams.max_seq_len = 131072                        
load_ne_hparams  9.hparams.alibi_bias_max = 0.000                         
load_ne_hparams  10.hparams.clip_qkv = 0.000                         
load_ne_hparams  11.hparams.par_res = 0                             
load_ne_hparams  12.hparams.word_embed_proj_dim = 0                             
load_ne_hparams  13.hparams.do_layer_norm_before = 0                             
load_ne_hparams  14.hparams.multi_query_group_num = 2                             
load_ne_hparams  15.hparams.ffn_hidden_size = 13696                         
load_ne_hparams  16.hparams.inner_hidden_size = 0                             
load_ne_hparams  17.hparams.n_experts = 0                             
load_ne_hparams  18.hparams.n_experts_used = 0                             
load_ne_hparams  19.hparams.n_embd_head_k = 0                             
load_ne_hparams  20.hparams.norm_eps = 0.000000                      
load_ne_hparams  21.hparams.freq_base = 5000000.000                   
load_ne_hparams  22.hparams.freq_scale = 1.000                         
load_ne_hparams  23.hparams.rope_scaling_factor = 0.000                         
load_ne_hparams  24.hparams.original_max_position_embeddings = 0                             
load_ne_hparams  25.hparams.use_yarn = 0                             
load_ne_vocab    26.vocab.bos_token_id = 1                             
load_ne_vocab    27.vocab.eos_token_id = 151329                        
load_ne_vocab    28.vocab.pad_token_id = 151329                        
load_ne_vocab    29.vocab.sep_token_id = -1                            
init: hparams.n_vocab         = 151552
init: hparams.n_embd          = 4096
init: hparams.n_mult          = 0
init: hparams.n_head          = 32
init: hparams.n_layer         = 40
init: hparams.n_rot           = 0
init: hparams.ffn_hidden_size = 13696
init: n_parts    = 1
load: ctx size   = 16528.38 MB
load: layers[0].ffn_fusion    = 1
load: scratch0   = 4096.00 MB
load: scratch1   = 2048.00 MB
load: scratch2   = 4096.00 MB
load: mem required  = 26768.38 MB (+ memory per state)
.............................................................................................
model_init_from_file: support_bestla_kv = 1
kv_cache_init: run_mha_reordered = 1
model_init_from_file: kv self size =  690.00 MB
Assistant:
你好👋！我是人工智能助手，很高兴为你服务。有什么可以帮助你的吗？
```

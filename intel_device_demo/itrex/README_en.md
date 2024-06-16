
# Using Intel® Extension for Transformers to Inference the GLM-4-9B-Chat Model

This example introduces how to use Intel® Extension for Transformers to inference the GLM-4-9B-Chat model.

## Device and Dependency Check

### Relevant Inference Test Data

**The data in this document is tested on the following hardware environment. The actual running environment requirements and memory usage may vary slightly. Please refer to the actual running environment.**

Test hardware information:

+ OS: Ubuntu 22.04 (This tutorial must be executed in a Linux environment)
+ Memory: 512GB
+ Python: 3.10.12
+ CPU: Intel(R) Xeon(R) Platinum 8358 CPU / 12th Gen Intel i5-12400

## Installing Dependencies

Before starting the inference, please install the dependencies in `basic_demo`, and you need to install the dependencies in this directory:
```shell
pip install -r requirements.txt
```

## Running Model Inference

```shell
python itrex_cli_demo.py
```

If this is your first inference, there will be a process of converting model weights. The converted model weights are stored in the `runtime_outputs` folder, which will consume about `60G` of disk space.
After the conversion is completed, there are two files in the folder:
+ ne_chatglm2_f32.bin 52G (If you do not use FP32 for inference, you can delete this file)
+ ne_chatglm2_q_nf4_bestla_cfp32_sym_sfp32_g32.bin 8.1G

If this is not your first inference, this step will be skipped, and you will directly start the conversation. The inference result is as follows:
```shell
Welcome to the CLI chat. Type your messages below.

User: Hello
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
load_ne_hparams  11.hparams.multi_query_group_num = 2                             
load_ne_hparams  12.hparams.ffn_hidden_size = 13696                         
load_ne_hparams  13.hparams.inner_hidden_size = 0                             
load_ne_hparams  14.hparams.n_experts = 0                             
load_ne_hparams  15.hparams.n_experts_used = 0                             
load_ne_hparams  16.hparams.n_embd_head_k = 0                             
load_ne_hparams  17.hparams.norm_eps = 0.000000                      
load_ne_hparams  18.hparams.freq_base = 5000000.000                   
load_ne_hparams  19.hparams.freq_scale = 1.000                         
load_ne_hparams  20.hparams.rope_scaling_factor = 0.000                         
load_ne_hparams  21.hparams.original_max_position_embeddings = 0                             
load_ne_hparams  22.hparams.use_yarn = 0                             
load_ne_vocab    23.vocab.bos_token_id = 1                             
load_ne_vocab    24.vocab.eos_token_id = 151329                        
load_ne_vocab    25.vocab.pad_token_id = 151329                        
load_ne_vocab    26.vocab.sep_token_id = -1                            
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
Hello👋! I am an AI assistant. How can I help you today?
```


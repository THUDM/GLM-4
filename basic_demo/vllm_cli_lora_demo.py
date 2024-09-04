"""
This script creates a CLI demo that utilizes LoRA adapters with vLLM backend for the GLM-4-9b model,
allowing users to interact with the model through a command-line interface.

Usage:
- Run the script to start the CLI demo.
- Interact with the model by typing questions and receiving responses.

Note: The script includes a modification to handle markdown to plain text conversion,
ensuring that the CLI interface displays formatted text correctly.
"""
import time
import asyncio
from transformers import AutoTokenizer
from vllm import SamplingParams, AsyncEngineArgs, AsyncLLMEngine
from typing import List, Dict
from vllm.lora.request import LoRARequest

MODEL_PATH = 'THUDM/GLM-4'
LORA_PATH = '' # 你的 lora adapter 路径

def load_model_and_tokenizer(model_dir: str):
    engine_args = AsyncEngineArgs(
        model=model_dir,
        tokenizer=model_dir,
        enable_lora=True, # 新增
        max_loras=1, # 新增
        max_lora_rank=8, ## 新增
        max_num_seqs=256, ## 新增
        tensor_parallel_size=2,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=0.5,
        max_model_len=2048,
        enforce_eager=True,
        worker_use_ray=True,
        engine_use_ray=False,
        disable_log_requests=True
        # 如果遇见 OOM 现象，建议开启下述参数
        # enable_chunked_prefill=True,
        # max_num_batched_tokens=8192
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True,
        encode_special_tokens=True
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    return engine, tokenizer


engine, tokenizer = load_model_and_tokenizer(MODEL_PATH)


async def vllm_gen(lora_path: str, messages: List[Dict[str, str]], top_p: float, temperature: float, max_dec_len: int):
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )
    params_dict = {
        "n": 1,
        "best_of": 1,
        "presence_penalty": 1.0,
        "frequency_penalty": 0.0,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": -1,
        "use_beam_search": False,
        "length_penalty": 1,
        "early_stopping": False,
        "ignore_eos": False,
        "max_tokens": max_dec_len,
        "logprobs": None,
        "prompt_logprobs": None,
        "skip_special_tokens": True
    }
    sampling_params = SamplingParams(**params_dict)
    async for output in engine.generate(inputs=inputs, sampling_params=sampling_params, request_id=f"{time.time()}", lora_request=LoRARequest("glm-4-lora", 1, lora_path=lora_path)):
        yield output.outputs[0].text


async def chat():
    history = []
    max_length = 8192
    top_p = 0.8
    temperature = 0

    print("Welcome to the GLM-4-9B CLI (Lora) chat. Type your messages below.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        history.append([user_input, ""])

        messages = []
        for idx, (user_msg, model_msg) in enumerate(history):
            if idx == len(history) - 1 and not model_msg:
                messages.append({"role": "user", "content": user_msg})
                break
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if model_msg:
                messages.append({"role": "assistant", "content": model_msg})

        print("\nGLM-4: ", end="")
        current_length = 0
        output = ""
        async for output in vllm_gen(LORA_PATH, messages, top_p, temperature, max_length):
            print(output[current_length:], end="", flush=True)
            current_length = len(output)
        history[-1][1] = output


if __name__ == "__main__":
    asyncio.run(chat())
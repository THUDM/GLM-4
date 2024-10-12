"""
This script creates a CLI demo with vllm backand for the glm-4v-9b model,
allowing users to interact with the model through a command-line interface.

Usage:
- Run the script to start the CLI demo.
- Interact with the model by typing questions and receiving responses.

Note: The script includes a modification to handle markdown to plain text conversion,
ensuring that the CLI interface displays formatted text correctly.
"""
import time
import asyncio
from PIL import Image
from typing import List, Dict
from vllm import SamplingParams, AsyncEngineArgs, AsyncLLMEngine

MODEL_PATH = 'THUDM/glm-4v-9b'

def load_model_and_tokenizer(model_dir: str):
    engine_args = AsyncEngineArgs(
        model=model_dir,
        tensor_parallel_size=1,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        worker_use_ray=True,
        disable_log_requests=True
        # 如果遇见 OOM 现象，建议开启下述参数
        # enable_chunked_prefill=True,
        # max_num_batched_tokens=8192
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    return engine

engine = load_model_and_tokenizer(MODEL_PATH)

async def vllm_gen(messages: List[Dict[str, str]], top_p: float, temperature: float, max_dec_len: int):
    inputs = messages[-1]
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
        "skip_special_tokens": True,
        "stop_token_ids" :[151329, 151336, 151338]
    }
    sampling_params = SamplingParams(**params_dict)

    async for output in engine.generate(inputs=inputs, sampling_params=sampling_params, request_id=f"{time.time()}"):
        yield output.outputs[0].text


async def chat():
    history = []
    max_length = 8192
    top_p = 0.8
    temperature = 0.6
    image = None

    print("Welcome to the GLM-4v-9B CLI chat. Type your messages below.")
    image_path = input("Image Path:")
    try:
        image = Image.open(image_path).convert("RGB")
    except:
        print("Invalid image path. Continuing with text conversation.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        history.append([user_input, ""])

        messages = []
        for idx, (user_msg, model_msg) in enumerate(history):
            if idx == len(history) - 1 and not model_msg:
                messages.append({
                    "prompt": user_msg,
                    "multi_modal_data": {
                        "image": image
                        },})
                break
            if user_msg:
                messages.append({"role": "user", "prompt": user_msg})
            if model_msg:
                messages.append({"role": "assistant", "prompt": model_msg})

        print("\nGLM-4v: ", end="")
        current_length = 0
        output = ""
        async for output in vllm_gen(messages, top_p, temperature, max_length):
            print(output[current_length:], end="", flush=True)
            current_length = len(output)
        history[-1][1] = output

if __name__ == "__main__":
    asyncio.run(chat())

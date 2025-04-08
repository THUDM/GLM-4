"""
This script creates a CLI demo with vllm backand for the glm-4v-9b model,
allowing users to interact with the model through a command-line interface.

Usage:
- Run the script to start the CLI demo.
- Interact with the model by typing questions and receiving responses.

Note: The script includes a modification to handle markdown to plain text conversion,
ensuring that the CLI interface displays formatted text correctly.
"""

import asyncio
import time
from typing import Dict, List

from PIL import Image
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams


MODEL_PATH = "THUDM/glm-4v-9b"


def load_model_and_tokenizer(model_dir: str):
    engine_args = AsyncEngineArgs(
        model=model_dir,
        tokenizer=model_dir,
        tensor_parallel_size=1,
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        disable_log_requests=True,
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
        "max_tokens": max_dec_len,
        "skip_special_tokens": True,
    }
    sampling_params = SamplingParams(**params_dict)

    async for output in engine.generate(prompt=inputs, sampling_params=sampling_params, request_id=f"{time.time()}"):
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
                messages.append(
                    {
                        "prompt": user_msg,
                        "multi_modal_data": {"image": image},
                    }
                )
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

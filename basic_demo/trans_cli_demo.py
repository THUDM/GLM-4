"""
This script creates a CLI demo with transformers backend for the glm-4-9b model,
allowing users to interact with the model through a command-line interface.

Usage:
- Run the script to start the CLI demo.
- Interact with the model by typing questions and receiving responses.

Note: The script includes a modification to handle markdown to plain text conversion,
ensuring that the CLI interface displays formatted text correctly.

If you use flash attention, you should install the flash-attn and  add attn_implementation="flash_attention_2" in model loading.
"""

import os
import torch
from threading import Thread
from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer, AutoModel

MODEL_PATH = os.environ.get('MODEL_PATH', 'THUDM/glm-4-9b-chat')

## If use peft model.
# def load_model_and_tokenizer(model_dir, trust_remote_code: bool = True):
#     if (model_dir / 'adapter_config.json').exists():
#         model = AutoModel.from_pretrained(
#             model_dir, trust_remote_code=trust_remote_code, device_map='auto'
#         )
#         tokenizer_dir = model.peft_config['default'].base_model_name_or_path
#     else:
#         model = AutoModel.from_pretrained(
#             model_dir, trust_remote_code=trust_remote_code, device_map='auto'
#         )
#         tokenizer_dir = model_dir
#     tokenizer = AutoTokenizer.from_pretrained(
#         tokenizer_dir, trust_remote_code=trust_remote_code, use_fast=False
#     )
#     return model, tokenizer


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH,trust_remote_code=True)

model = AutoModel.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    # attn_implementation="flash_attention_2", # Use Flash Attention
    # torch_dtype=torch.bfloat16, #using flash-attn must use bfloat16 or float16
    device_map="auto").eval()


class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = model.config.eos_token_id
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


if __name__ == "__main__":
    history = []
    max_length = 8192
    top_p = 0.8
    temperature = 0.6
    stop = StopOnTokens()

    print("Welcome to the GLM-4-9B CLI chat. Type your messages below.")
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
        model_inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt"
        ).to(model.device)
        streamer = TextIteratorStreamer(
            tokenizer=tokenizer,
            timeout=60,
            skip_prompt=True,
            skip_special_tokens=True
        )
        generate_kwargs = {
            "input_ids": model_inputs,
            "streamer": streamer,
            "max_new_tokens": max_length,
            "do_sample": True,
            "top_p": top_p,
            "temperature": temperature,
            "stopping_criteria": StoppingCriteriaList([stop]),
            "repetition_penalty": 1.2,
            "eos_token_id": model.config.eos_token_id,
        }
        t = Thread(target=model.generate, kwargs=generate_kwargs)
        t.start()
        print("GLM-4:", end="", flush=True)
        for new_token in streamer:
            if new_token:
                print(new_token, end="", flush=True)
                history[-1][1] += new_token

        history[-1][1] = history[-1][1].strip()

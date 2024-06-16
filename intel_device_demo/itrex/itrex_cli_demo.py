"""
This script creates a CLI demo with transformers backend for the glm-4-9b model with Intel® Extension for Transformers
"""

import os
MODEL_PATH = os.environ.get('MODEL_PATH', 'THUDM/glm-4-9b-chat')

import torch
from threading import Thread
from intel_extension_for_transformers.transformers import AutoModelForCausalLM
from transformers import TextIteratorStreamer, StoppingCriteriaList, StoppingCriteria, AutoTokenizer


class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [151329, 151336, 151338]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


def initialize_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="cpu", # Use Intel CPU for inference
        trust_remote_code=True,
        load_in_4bit=True
    )
    return tokenizer, model


def get_user_input():
    return input("\nUser: ")


def main():
    tokenizer, model = initialize_model_and_tokenizer()

    history = []
    max_length = 100
    top_p = 0.9
    temperature = 0.8
    stop = StopOnTokens()

    print("Welcome to the CLI chat. Type your messages below.")
    while True:
        user_input = get_user_input()
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
        )

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
        print("Assistant:", end="", flush=True)
        for new_token in streamer:
            if new_token:
                print(new_token, end="", flush=True)
                history[-1][1] += new_token

        history[-1][1] = history[-1][1].strip()


if __name__ == "__main__":
    main()

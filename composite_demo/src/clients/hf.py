"""
HuggingFace client.
"""

import threading
from collections.abc import Generator
from threading import Thread

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

from client import Client, process_input, process_response
from conversation import Conversation


class HFClient(Client):
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        ).eval()

    def generate_stream(
        self,
        tools: list[dict],
        history: list[Conversation],
        **parameters,
    ) -> Generator[tuple[str | dict, list[dict]]]:
        chat_history = process_input(history, tools)
        model_inputs = self.tokenizer.apply_chat_template(
            chat_history,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)
        streamer = TextIteratorStreamer(
            tokenizer=self.tokenizer,
            timeout=5,
            skip_prompt=True,
        )
        generate_kwargs = {
            **model_inputs,
            "streamer": streamer,
            "eos_token_id": [151329, 151336, 151338],
            "do_sample": True,
        }
        generate_kwargs.update(parameters)
        t = Thread(target=self.model.generate, kwargs=generate_kwargs)
        t.start()
        total_text = ""
        for token_text in streamer:
            total_text += token_text
            yield process_response(total_text, chat_history)

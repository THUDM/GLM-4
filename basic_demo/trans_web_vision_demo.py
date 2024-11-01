"""
This script creates a Gradio demo with a Transformers backend for the glm-4v-9b model, allowing users to interact with the model through a Gradio web UI.

Usage:
- Run the script to start the Gradio server.
- Interact with the model via the web UI.

Requirements:
- Gradio package
  - Type `pip install gradio==4.44.1` to install Gradio.
"""

import os
import torch
import gradio as gr
from threading import Thread
from transformers import (
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer, AutoModel
)
from PIL import Image
import requests
from io import BytesIO

MODEL_PATH = os.environ.get('MODEL_PATH', 'THUDM/glm-4v-9b')

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    encode_special_tokens=True
)
model = AutoModel.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.bfloat16
).eval()

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = model.config.eos_token_id
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

def get_image(image_path=None, image_url=None):
    if image_path:
        return Image.open(image_path).convert("RGB")
    elif image_url:
        response = requests.get(image_url)
        return Image.open(BytesIO(response.content)).convert("RGB")
    return None

def chatbot(image_path=None, image_url=None, assistant_prompt=""):
    image = get_image(image_path, image_url)

    messages = [
        {"role": "assistant", "content": assistant_prompt},
        {"role": "user", "content": "", "image": image}
    ]

    model_inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True
    ).to(next(model.parameters()).device)

    streamer = TextIteratorStreamer(
        tokenizer=tokenizer,
        timeout=60,
        skip_prompt=True,
        skip_special_tokens=True
    )

    generate_kwargs = {
        **model_inputs,
        "streamer": streamer,
        "max_new_tokens": 1024,
        "do_sample": True,
        "top_p": 0.8,
        "temperature": 0.6,
        "stopping_criteria": StoppingCriteriaList([StopOnTokens()]),
        "repetition_penalty": 1.2,
        "eos_token_id": [151329, 151336, 151338],
    }

    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    response = ""
    for new_token in streamer:
        if new_token:
            response += new_token

    return image, response.strip()

with gr.Blocks() as demo:
    demo.title = "GLM-4V-9B Image Recognition Demo"
    demo.description = """
    This demo uses the GLM-4V-9B model to got image infomation.
    """
    with gr.Row():
        with gr.Column():
            image_path_input = gr.File(label="Upload Image (High-Priority)", type="filepath")
            image_url_input = gr.Textbox(label="Image URL (Low-Priority)")
            assistant_prompt_input = gr.Textbox(label="Assistant Prompt (You Can Change It)", value="这是什么？")
            submit_button = gr.Button("Submit")
        with gr.Column():
            chatbot_output = gr.Textbox(label="GLM-4V-9B Model Response")
            image_output = gr.Image(label="Image Preview")

    submit_button.click(chatbot,
                        inputs=[image_path_input, image_url_input, assistant_prompt_input],
                        outputs=[image_output, chatbot_output])

demo.launch(server_name="127.0.0.1", server_port=8911, inbrowser=True, share=False)
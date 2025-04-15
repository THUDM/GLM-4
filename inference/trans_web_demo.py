from threading import Thread

import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


MODEL_PATH = "THUDM/GLM-4-9B-0414"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto")


def preprocess_messages(history, system_prompt):
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    for idx, (user_msg, model_msg) in enumerate(history):
        if idx == len(history) - 1 and not model_msg:
            messages.append({"role": "user", "content": user_msg})
            break
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if model_msg:
            messages.append({"role": "assistant", "content": model_msg})

    return messages


def predict(history, system_prompt, max_length, top_p, top_k, temperature):
    messages = preprocess_messages(history, system_prompt)
    model_inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    streamer = TextIteratorStreamer(tokenizer, timeout=60, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = {
        "input_ids": model_inputs["input_ids"],
        "attention_mask": model_inputs["attention_mask"],
        "streamer": streamer,
        "max_new_tokens": max_length,
        "do_sample": True,
        "top_p": top_p,
        "top_k": top_k,
        "temperature": temperature,
        "repetition_penalty": 1.2,
    }

    generate_kwargs["eos_token_id"] = tokenizer.encode("<|user|>")

    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()
    for new_token in streamer:
        if new_token:
            history[-1][1] += new_token
        yield history


def main():
    with gr.Blocks() as demo:
        gr.HTML("""<h1 align="center">GLM-4-0414 Gradio Demo</h1>""")

        with gr.Row():
            with gr.Column(scale=3):
                system_prompt = gr.Textbox(
                    show_label=True, placeholder="Enter system prompt here...", label="System Prompt", lines=2
                )

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot()

        with gr.Row():
            with gr.Column(scale=2):
                user_input = gr.Textbox(show_label=True, placeholder="Input...", label="User Input")
                submitBtn = gr.Button("Submit")
                emptyBtn = gr.Button("Clear History")
            with gr.Column(scale=1):
                max_length = gr.Slider(0, 8192, value=4096, step=1.0, label="Maximum length", interactive=True)
                top_p = gr.Slider(0, 1, value=0.8, step=0.01, label="Top P", interactive=True)
                top_k = gr.Slider(0, 100, value=50, step=1, label="Top K", interactive=True)
                temperature = gr.Slider(0.01, 1, value=0.6, step=0.01, label="Temperature", interactive=True)

        def user(query, history):
            return "", history + [[query, ""]]

        def clear_history():
            return None

        submitBtn.click(user, [user_input, chatbot], [user_input, chatbot], queue=False).then(
            predict, [chatbot, system_prompt, max_length, top_p, top_k, temperature], chatbot
        )
        emptyBtn.click(clear_history, None, [chatbot], queue=False)

    demo.queue()
    demo.launch()


if __name__ == "__main__":
    main()

import argparse
from threading import Thread
from typing import List, Tuple

import torch
from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoConfig, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer


class StopOnTokens(StoppingCriteria):
    def __init__(self, token_ids):
        self.token_ids = token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in self.token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-h", "--help", action="help", help="Show this help message and exit.")
    parser.add_argument("-m", "--model_path", required=True, type=str, help="Required. model path")
    parser.add_argument(
        "-l", "--max_sequence_length", default=256, required=False, type=int, help="Required. maximun length of output"
    )
    parser.add_argument(
        "-d", "--device", default="CPU", required=False, type=str, help="Required. device for inference"
    )
    args = parser.parse_args()
    model_dir = args.model_path

    ov_config = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": ""}

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    print("====Compiling model====")
    ov_model = OVModelForCausalLM.from_pretrained(
        model_dir,
        device=args.device,
        ov_config=ov_config,
        config=AutoConfig.from_pretrained(model_dir, trust_remote_code=True),
        trust_remote_code=True,
    )

    streamer = TextIteratorStreamer(tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)
    stop_tokens = [StopOnTokens([151329, 151336, 151338])]

    def convert_history_to_token(history: List[Tuple[str, str]]):
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
            messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
        )
        return model_inputs

    history = []
    print("====Starting conversation====")
    while True:
        input_text = input("用户: ")
        if input_text.lower() == "stop":
            break

        if input_text.lower() == "clear":
            history = []
            print("AI助手: 对话历史已清空")
            continue

        print("GLM-4-9B-OpenVINO:", end=" ")
        history = history + [[input_text, ""]]
        model_inputs = convert_history_to_token(history)
        generate_kwargs = dict(
            input_ids=model_inputs,
            max_new_tokens=args.max_sequence_length,
            temperature=0.1,
            do_sample=True,
            top_p=1.0,
            top_k=50,
            repetition_penalty=1.1,
            streamer=streamer,
            stopping_criteria=StoppingCriteriaList(stop_tokens),
        )

        t1 = Thread(target=ov_model.generate, kwargs=generate_kwargs)
        t1.start()

        partial_text = ""
        for new_text in streamer:
            new_text = new_text
            print(new_text, end="", flush=True)
            partial_text += new_text
        print("\n")
        history[-1][1] = partial_text

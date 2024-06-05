from pathlib import Path
from typing import Annotated, Union

import typer
from peft import AutoPeftModelForCausalLM, PeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast
)

ModelType = Union[PreTrainedModel, PeftModelForCausalLM]
TokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

app = typer.Typer(pretty_exceptions_show_locals=False)


def load_model_and_tokenizer(
        model_dir: Union[str, Path], trust_remote_code: bool = True
) -> tuple[ModelType, TokenizerType]:
    model_dir = Path(model_dir).expanduser().resolve()
    if (model_dir / 'adapter_config.json').exists():
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=trust_remote_code, device_map='auto'
        )
        tokenizer_dir = model.peft_config['default'].base_model_name_or_path
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=trust_remote_code, device_map='auto'
        )
        tokenizer_dir = model_dir
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir, trust_remote_code=trust_remote_code, encode_special_tokens=True, use_fast=False
    )
    return model, tokenizer


@app.command()
def main(
        model_dir: Annotated[str, typer.Argument(help='')],
):
    messages = [
        {
            "role": "system", "content": "",
            "tools":
                [
                    {
                        "type": "function",
                        "function": {
                            "name": "create_calendar_event",
                            "description": "Create a new calendar event",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "title": {
                                        "type": "string",
                                        "description": "The title of the event"
                                    },
                                    "start_time": {
                                        "type": "string",
                                        "description": "The start time of the event in the format YYYY-MM-DD HH:MM"
                                    },
                                    "end_time": {
                                        "type": "string",
                                        "description": "The end time of the event in the format YYYY-MM-DD HH:MM"
                                    }
                                },
                                "required": [
                                    "title",
                                    "start_time",
                                    "end_time"
                                ]
                            }
                        }
                    }
                ]

        },
        {
            "role": "user",
            "content": "Can you help me create a calendar event for my meeting tomorrow? The title is \"Team Meeting\". It starts at 10:00 AM and ends at 11:00 AM."
        },
    ]
    model, tokenizer = load_model_and_tokenizer(model_dir)
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt"
    ).to(model.device)
    generate_kwargs = {
        "input_ids": inputs,
        "max_new_tokens": 1024,
        "do_sample": True,
        "top_p": 0.8,
        "temperature": 0.8,
        "repetition_penalty": 1.2,
        "eos_token_id": model.config.eos_token_id,
    }
    outputs = model.generate(**generate_kwargs)
    response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True).strip()
    print("=========")
    print(response)


if __name__ == '__main__':
    app()

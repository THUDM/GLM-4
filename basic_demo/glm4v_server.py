import gc
import threading
import time
import base64
import sys
from contextlib import asynccontextmanager
from typing import List, Literal, Union, Tuple, Optional
import torch
import uvicorn
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from transformers import (
    AutoTokenizer,
    AutoModel,
    TextIteratorStreamer
)
from peft import PeftModelForCausalLM
from PIL import Image
from io import BytesIO
from pathlib import Path

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TORCH_TYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    An asynchronous context manager for managing the lifecycle of the FastAPI app.
    It ensures that GPU memory is cleared after the app's lifecycle ends, which is essential for efficient resource management in GPU environments.
    """
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ModelCard(BaseModel):
    """
    A Pydantic model representing a model card, which provides metadata about a machine learning model.
    It includes fields like model ID, owner, and creation time.
    """
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "owner"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []


class ImageUrl(BaseModel):
    url: str


class TextContent(BaseModel):
    type: Literal["text"]
    text: str


class ImageUrlContent(BaseModel):
    type: Literal["image_url"]
    image_url: ImageUrl


ContentItem = Union[TextContent, ImageUrlContent]


class ChatMessageInput(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: Union[str, List[ContentItem]]
    name: Optional[str] = None


class ChatMessageResponse(BaseModel):
    role: Literal["assistant"]
    content: str = None
    name: Optional[str] = None


class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessageInput]
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.8
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    # Additional parameters
    repetition_penalty: Optional[float] = 1.0


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessageResponse


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


class ChatCompletionResponse(BaseModel):
    model: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
    usage: Optional[UsageInfo] = None


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    """
    An endpoint to list available models. It returns a list of model cards.
    This is useful for clients to query and understand what models are available for use.
    """
    model_card = ModelCard(id="GLM-4v-9b")
    return ModelList(data=[model_card])


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    global model, tokenizer

    if len(request.messages) < 1 or request.messages[-1].role == "assistant":
        raise HTTPException(status_code=400, detail="Invalid request")

    gen_params = dict(
        messages=request.messages,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens or 1024,
        echo=False,
        stream=request.stream,
        repetition_penalty=request.repetition_penalty
    )

    if request.stream:
        generate = predict(request.model, gen_params)
        return EventSourceResponse(generate, media_type="text/event-stream")
    response = generate_glm4v(model, tokenizer, gen_params)

    usage = UsageInfo()

    message = ChatMessageResponse(
        role="assistant",
        content=response["text"],
    )
    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=message,
    )
    task_usage = UsageInfo.model_validate(response["usage"])
    for usage_key, usage_value in task_usage.model_dump().items():
        setattr(usage, usage_key, getattr(usage, usage_key) + usage_value)
    return ChatCompletionResponse(model=request.model, choices=[choice_data], object="chat.completion", usage=usage)


def predict(model_id: str, params: dict):
    global model, tokenizer

    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(role="assistant"),
        finish_reason=None
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))

    previous_text = ""
    for new_response in generate_stream_glm4v(model, tokenizer, params):
        decoded_unicode = new_response["text"]
        delta_text = decoded_unicode[len(previous_text):]
        previous_text = decoded_unicode
        delta = DeltaMessage(content=delta_text, role="assistant")
        choice_data = ChatCompletionResponseStreamChoice(index=0, delta=delta)
        chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
        yield "{}".format(chunk.model_dump_json(exclude_unset=True))

    choice_data = ChatCompletionResponseStreamChoice(index=0, delta=DeltaMessage())
    chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))


def generate_glm4v(model: AutoModel, tokenizer: AutoTokenizer, params: dict):
    """
    Generates a response using the GLM-4v-9b model. It processes the chat history and image data, if any,
    and then invokes the model to generate a response.
    """

    response = None

    for response in generate_stream_glm4v(model, tokenizer, params):
        pass
    return response


def process_history_and_images(messages: List[ChatMessageInput]) -> Tuple[
    Optional[str], Optional[List[Tuple[str, str]]], Optional[List[Image.Image]]]:
    """
    Process history messages to extract text, identify the last user query,
    and convert base64 encoded image URLs to PIL images.

    Args:
        messages(List[ChatMessageInput]): List of ChatMessageInput objects.
    return: A tuple of three elements:
             - The last user query as a string.
             - Text history formatted as a list of tuples for the model.
             - List of PIL Image objects extracted from the messages.
    """

    formatted_history = []
    image_list = []
    last_user_query = ''

    for i, message in enumerate(messages):
        role = message.role
        content = message.content

        if isinstance(content, list):  # text
            text_content = ' '.join(item.text for item in content if isinstance(item, TextContent))
        else:
            text_content = content

        if isinstance(content, list):  # image
            for item in content:
                if isinstance(item, ImageUrlContent):
                    image_url = item.image_url.url
                    if image_url.startswith("data:image/jpeg;base64,"):
                        base64_encoded_image = image_url.split("data:image/jpeg;base64,")[1]
                        image_data = base64.b64decode(base64_encoded_image)
                        image = Image.open(BytesIO(image_data)).convert('RGB')
                    else:
                        response = requests.get(image_url, verify=False)
                        image = Image.open(BytesIO(response.content)).convert('RGB')
                    image_list.append(image)

        if role == 'user':
            if i == len(messages) - 1:  # 最后一条用户消息
                last_user_query = text_content
            else:
                formatted_history.append((text_content, ''))
        elif role == 'assistant':
            if formatted_history:
                if formatted_history[-1][1] != '':
                    assert False, f"the last query is answered. answer again. {formatted_history[-1][0]}, {formatted_history[-1][1]}, {text_content}"
                formatted_history[-1] = (formatted_history[-1][0], text_content)
            else:
                assert False, f"assistant reply before user"
        else:
            assert False, f"unrecognized role: {role}"

    return last_user_query, formatted_history, image_list


@torch.inference_mode()
def generate_stream_glm4v(model: AutoModel, tokenizer: AutoTokenizer, params: dict):
    messages = params["messages"]
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    max_new_tokens = int(params.get("max_tokens", 256))
    query, history, image_list = process_history_and_images(messages)

    inputs = []
    for idx, (user_msg, model_msg) in enumerate(history):
        if idx == len(history) - 1 and not model_msg:
            inputs.append({"role": "user", "content": user_msg})
            if image_list and not uploaded:
                inputs[-1].update({"image": image_list[0]})
                uploaded = True
            break
        if user_msg:
            inputs.append({"role": "user", "content": user_msg})
        if model_msg:
            inputs.append({"role": "assistant", "content": model_msg})
    if len(image_list) >= 1:
        inputs.append({"role": "user", "content": query, "image": image_list[0]})
    else:
        inputs.append({"role": "user", "content": query})

    model_inputs = tokenizer.apply_chat_template(
        inputs,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True
    ).to(next(model.parameters()).device)
    input_echo_len = len(model_inputs["input_ids"][0])
    streamer = TextIteratorStreamer(
        tokenizer=tokenizer,
        timeout=60.0,
        skip_prompt=True,
        skip_special_tokens=True
    )
    gen_kwargs = {
        "repetition_penalty": repetition_penalty,
        "max_new_tokens": max_new_tokens,
        "do_sample": True if temperature > 1e-5 else False,
        "top_p": top_p if temperature > 1e-5 else 0,
        "top_k": 1,
        'streamer': streamer,
    }
    if temperature > 1e-5:
        gen_kwargs["temperature"] = temperature

    generated_text = ""

    def generate_text():
        with torch.no_grad():
            model.generate(**model_inputs, **gen_kwargs)

    generation_thread = threading.Thread(target=generate_text)
    generation_thread.start()

    total_len = input_echo_len
    for next_text in streamer:
        generated_text += next_text
        total_len = len(tokenizer.encode(generated_text))
        yield {
            "text": generated_text,
            "usage": {
                "prompt_tokens": input_echo_len,
                "completion_tokens": total_len - input_echo_len,
                "total_tokens": total_len,
            },
        }
    generation_thread.join()

    yield {
        "text": generated_text,
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": total_len - input_echo_len,
            "total_tokens": total_len,
        },
    }


gc.collect()
torch.cuda.empty_cache()

if __name__ == "__main__":
    MODEL_PATH = sys.argv[1]
    model_dir = Path(MODEL_PATH).expanduser().resolve()
    if (model_dir / 'adapter_config.json').exists():
        import json
        with open(model_dir / 'adapter_config.json', 'r', encoding='utf-8') as file:
            config = json.load(file)
        model = AutoModel.from_pretrained(
            config.get('base_model_name_or_path'),
            trust_remote_code=True,
            device_map='auto',
            torch_dtype=TORCH_TYPE
        )
        model = PeftModelForCausalLM.from_pretrained(
            model=model,
            model_id=model_dir,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            config.get('base_model_name_or_path'),
            trust_remote_code=True,
            encode_special_tokens=True
        )
        model.eval().to(DEVICE)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        encode_special_tokens=True
        )
        model = AutoModel.from_pretrained(
            MODEL_PATH,
            torch_dtype=TORCH_TYPE,
            trust_remote_code=True,
            device_map="auto",
        ).eval().to(DEVICE)

    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)

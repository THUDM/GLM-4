import base64
import binascii
import gc
import ipaddress
import os
import secrets
import socket
import sys
import threading
import time
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union
from urllib.parse import urljoin, urlparse

import requests
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from peft import PeftModelForCausalLM
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field, field_validator, model_validator
from sse_starlette.sse import EventSourceResponse
from starlette.responses import JSONResponse
from transformers import AutoModel, AutoTokenizer, TextIteratorStreamer


TORCH_TYPE = (
    torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
)

MAX_REQUEST_BYTES = int(os.environ.get("GLM4V_MAX_REQUEST_BYTES", 30 * 1024 * 1024))
MAX_IMAGE_BYTES = int(os.environ.get("GLM4V_MAX_IMAGE_BYTES", 20 * 1024 * 1024))
MAX_IMAGE_PIXELS = int(os.environ.get("GLM4V_MAX_IMAGE_PIXELS", 20_000_000))
MAX_MESSAGES = int(os.environ.get("GLM4V_MAX_MESSAGES", 64))
MAX_TEXT_CHARS = int(os.environ.get("GLM4V_MAX_TEXT_CHARS", 200_000))
MAX_OUTPUT_TOKENS = int(os.environ.get("GLM4V_MAX_OUTPUT_TOKENS", 4096))
MAX_REDIRECTS = int(os.environ.get("GLM4V_MAX_IMAGE_REDIRECTS", 3))
IMAGE_CONNECT_TIMEOUT = float(os.environ.get("GLM4V_IMAGE_CONNECT_TIMEOUT", 3))
IMAGE_READ_TIMEOUT = float(os.environ.get("GLM4V_IMAGE_READ_TIMEOUT", 10))
API_KEY = os.environ.get("GLM4V_API_KEY")
ALLOWED_IMAGE_HOSTS = {
    host.strip().rstrip(".").lower()
    for host in os.environ.get("GLM4V_ALLOWED_IMAGE_HOSTS", "").split(",")
    if host.strip()
}
INFERENCE_SLOTS = threading.BoundedSemaphore(int(os.environ.get("GLM4V_MAX_CONCURRENT_REQUESTS", 1)))


class RequestTooLarge(Exception):
    pass


class RequestBodyLimitMiddleware:
    """Reject oversized request bodies, including chunked bodies, before parsing."""

    def __init__(self, app, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        content_length = headers.get(b"content-length")
        if content_length:
            try:
                parsed_length = int(content_length)
                if parsed_length < 0:
                    raise ValueError
                if parsed_length > self.max_bytes:
                    response = JSONResponse({"detail": "Request body too large"}, status_code=413)
                    await response(scope, receive, send)
                    return
            except ValueError:
                response = JSONResponse({"detail": "Invalid Content-Length"}, status_code=400)
                await response(scope, receive, send)
                return

        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise RequestTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except RequestTooLarge:
            response = JSONResponse({"detail": "Request body too large"}, status_code=413)
            await response(scope, receive, send)


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

app.add_middleware(RequestBodyLimitMiddleware, max_bytes=MAX_REQUEST_BYTES)

cors_origins = [origin.strip() for origin in os.environ.get("GLM4V_CORS_ORIGINS", "").split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials="*" not in cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)


@app.middleware("http")
async def require_api_key(request: Request, call_next):
    if API_KEY:
        supplied = request.headers.get("Authorization", "")
        expected = f"Bearer {API_KEY}"
        if not secrets.compare_digest(supplied, expected):
            return JSONResponse({"detail": "Unauthorized"}, status_code=401, headers={"WWW-Authenticate": "Bearer"})
    else:
        client_host = request.client.host if request.client else ""
        try:
            is_loopback = ipaddress.ip_address(client_host).is_loopback
        except ValueError:
            is_loopback = client_host == "localhost"
        if not is_loopback:
            return JSONResponse({"detail": "API key is required for non-loopback clients"}, status_code=403)
    return await call_next(request)


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
    data: List[ModelCard] = Field(default_factory=list)


class ImageUrl(BaseModel):
    url: str = Field(min_length=1, max_length=((MAX_IMAGE_BYTES + 2) // 3) * 4 + 256)

    @field_validator("url")
    @classmethod
    def limit_remote_url_length(cls, value: str) -> str:
        if not value.startswith("data:image/") and len(value) > 4096:
            raise ValueError("Remote image URL exceeds 4096 characters")
        return value


class TextContent(BaseModel):
    type: Literal["text"]
    text: str = Field(max_length=MAX_TEXT_CHARS)


class ImageUrlContent(BaseModel):
    type: Literal["image_url"]
    image_url: ImageUrl


ContentItem = Union[TextContent, ImageUrlContent]


class ChatMessageInput(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: Union[str, List[ContentItem]]
    name: Optional[str] = Field(default=None, max_length=128)


class ChatMessageResponse(BaseModel):
    role: Literal["assistant"]
    content: str = None
    name: Optional[str] = None


class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = Field(min_length=1, max_length=128)
    messages: List[ChatMessageInput] = Field(min_length=1, max_length=MAX_MESSAGES)
    temperature: Optional[float] = Field(default=0.8, ge=0, le=2)
    top_p: Optional[float] = Field(default=0.8, ge=0, le=1)
    max_tokens: Optional[int] = Field(default=None, ge=1, le=MAX_OUTPUT_TOKENS)
    stream: Optional[bool] = False
    # Additional parameters
    repetition_penalty: Optional[float] = Field(default=1.0, ge=0.1, le=2)

    @model_validator(mode="after")
    def enforce_content_limits(self):
        text_chars = 0
        image_count = 0
        for message in self.messages:
            if isinstance(message.content, str):
                text_chars += len(message.content)
                continue
            for item in message.content:
                if isinstance(item, TextContent):
                    text_chars += len(item.text)
                elif isinstance(item, ImageUrlContent):
                    image_count += 1
        if text_chars > MAX_TEXT_CHARS:
            raise ValueError(f"Total text content exceeds {MAX_TEXT_CHARS} characters")
        if image_count > 1:
            raise ValueError("GLM-4V accepts at most one image per request")
        return self


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
def create_chat_completion(request: ChatCompletionRequest):
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
        repetition_penalty=request.repetition_penalty,
    )

    if not INFERENCE_SLOTS.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="The model is busy; retry later")

    if request.stream:
        return EventSourceResponse(predict_with_slot(request.model, gen_params), media_type="text/event-stream")

    try:
        response = generate_glm4v(model, tokenizer, gen_params)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail="Unable to download remote image") from exc
    finally:
        INFERENCE_SLOTS.release()

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


def predict_with_slot(model_id: str, params: dict):
    try:
        yield from predict(model_id, params)
    finally:
        INFERENCE_SLOTS.release()


def predict(model_id: str, params: dict):
    global model, tokenizer

    choice_data = ChatCompletionResponseStreamChoice(index=0, delta=DeltaMessage(role="assistant"), finish_reason=None)
    chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))

    previous_text = ""
    for new_response in generate_stream_glm4v(model, tokenizer, params):
        decoded_unicode = new_response["text"]
        delta_text = decoded_unicode[len(previous_text) :]
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


def _validate_remote_image_url(url: str) -> None:
    if not ALLOWED_IMAGE_HOSTS:
        raise ValueError(
            "Remote image URLs are disabled. Set GLM4V_ALLOWED_IMAGE_HOSTS to an explicit comma-separated allowlist."
        )

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Image URL must use http or https")
    if parsed.username or parsed.password:
        raise ValueError("Credentials in image URLs are not allowed")

    hostname = parsed.hostname.rstrip(".").lower()
    if hostname not in ALLOWED_IMAGE_HOSTS:
        raise ValueError(f"Image host is not allowlisted: {hostname}")

    try:
        addresses = socket.getaddrinfo(hostname, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError(f"Unable to resolve image host: {hostname}") from exc
    if not addresses:
        raise ValueError(f"Unable to resolve image host: {hostname}")

    for _, _, _, _, sockaddr in addresses:
        address = ipaddress.ip_address(sockaddr[0].split("%", 1)[0])
        if not address.is_global:
            raise ValueError(f"Image host resolves to a non-public address: {address}")


def _load_image_bytes(image_data: bytes) -> Image.Image:
    if len(image_data) > MAX_IMAGE_BYTES:
        raise ValueError(f"Image exceeds the {MAX_IMAGE_BYTES}-byte limit")
    try:
        with Image.open(BytesIO(image_data)) as source:
            source.verify()
        with Image.open(BytesIO(image_data)) as source:
            width, height = source.size
            if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                raise ValueError(f"Image exceeds the {MAX_IMAGE_PIXELS}-pixel limit")
            return source.convert("RGB")
    except Image.DecompressionBombError as exc:
        raise ValueError("Image dimensions are unsafe") from exc
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Invalid image data") from exc


def _decode_data_image(image_url: str) -> Image.Image:
    header, separator, encoded = image_url.partition(",")
    if not separator or not header.startswith("data:image/") or not header.endswith(";base64"):
        raise ValueError("Only base64-encoded image data URLs are supported")
    if len(encoded) > ((MAX_IMAGE_BYTES + 2) // 3) * 4:
        raise ValueError(f"Encoded image exceeds the {MAX_IMAGE_BYTES}-byte limit")
    try:
        image_data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Invalid base64 image") from exc
    return _load_image_bytes(image_data)


def _download_remote_image(image_url: str) -> Image.Image:
    session = requests.Session()
    session.trust_env = False
    current_url = image_url
    try:
        for redirect_count in range(MAX_REDIRECTS + 1):
            _validate_remote_image_url(current_url)
            response = session.get(
                current_url,
                allow_redirects=False,
                headers={"Accept": "image/*", "User-Agent": "GLM-4V-image-fetcher/1.0"},
                stream=True,
                timeout=(IMAGE_CONNECT_TIMEOUT, IMAGE_READ_TIMEOUT),
                verify=True,
            )
            try:
                if response.is_redirect or response.is_permanent_redirect:
                    if redirect_count >= MAX_REDIRECTS:
                        raise ValueError("Too many image redirects")
                    location = response.headers.get("Location")
                    if not location:
                        raise ValueError("Image redirect has no Location header")
                    current_url = urljoin(current_url, location)
                    continue

                response.raise_for_status()
                content_type = response.headers.get("Content-Type", "").split(";", 1)[0].lower()
                if not content_type.startswith("image/"):
                    raise ValueError(f"Remote resource is not an image: {content_type or 'unknown content type'}")
                content_length = response.headers.get("Content-Length")
                if content_length:
                    try:
                        declared_size = int(content_length)
                    except ValueError as exc:
                        raise ValueError("Remote server returned an invalid Content-Length") from exc
                    if declared_size > MAX_IMAGE_BYTES:
                        raise ValueError(f"Remote image exceeds the {MAX_IMAGE_BYTES}-byte limit")

                chunks = []
                total = 0
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > MAX_IMAGE_BYTES:
                        raise ValueError(f"Remote image exceeds the {MAX_IMAGE_BYTES}-byte limit")
                    chunks.append(chunk)
                return _load_image_bytes(b"".join(chunks))
            finally:
                response.close()
    finally:
        session.close()
    raise ValueError("Unable to download image")


def process_history_and_images(
    messages: List[ChatMessageInput],
) -> Tuple[Optional[str], Optional[List[Tuple[str, str]]], Optional[List[Image.Image]]]:
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
    last_user_query = ""

    for i, message in enumerate(messages):
        role = message.role
        content = message.content

        if isinstance(content, list):  # text
            text_content = " ".join(item.text for item in content if isinstance(item, TextContent))
        else:
            text_content = content

        if isinstance(content, list):  # image
            for item in content:
                if isinstance(item, ImageUrlContent):
                    image_url = item.image_url.url
                    if image_url.startswith("data:image/"):
                        image = _decode_data_image(image_url)
                    else:
                        image = _download_remote_image(image_url)
                    image_list.append(image)

        if role == "user":
            if i == len(messages) - 1:  # 最后一条用户消息
                last_user_query = text_content
            else:
                formatted_history.append((text_content, ""))
        elif role == "assistant":
            if formatted_history:
                if formatted_history[-1][1] != "":
                    assert False, f"the last query is answered. answer again. {formatted_history[-1][0]}, {formatted_history[-1][1]}, {text_content}"
                formatted_history[-1] = (formatted_history[-1][0], text_content)
            else:
                assert False, "assistant reply before user"
        else:
            assert False, f"unrecognized role: {role}"

    return last_user_query, formatted_history, image_list


@torch.inference_mode()
def generate_stream_glm4v(model: AutoModel, tokenizer: AutoTokenizer, params: dict):
    uploaded = False
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
        inputs, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True
    ).to(next(model.parameters()).device)

    input_echo_len = len(model_inputs["input_ids"][0])
    streamer = TextIteratorStreamer(tokenizer=tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = {
        "repetition_penalty": repetition_penalty,
        "max_new_tokens": max_new_tokens,
        "do_sample": True if temperature > 1e-5 else False,
        "top_p": top_p if temperature > 1e-5 else 0,
        "top_k": 1,
        "streamer": streamer,
        "eos_token_id": [151329, 151336, 151338],
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
    print("\033[91m--generated_text\033[0m", generated_text)
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
    if (model_dir / "adapter_config.json").exists():
        import json

        with open(model_dir / "adapter_config.json", "r", encoding="utf-8") as file:
            config = json.load(file)
        model = AutoModel.from_pretrained(
            config.get("base_model_name_or_path"), device_map="auto", torch_dtype=TORCH_TYPE
        )
        model = PeftModelForCausalLM.from_pretrained(
            model=model,
            model_id=model_dir,
        )
        tokenizer = AutoTokenizer.from_pretrained(config.get("base_model_name_or_path"), encode_special_tokens=True)
        model.eval()
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, encode_special_tokens=True)
        model = AutoModel.from_pretrained(
            MODEL_PATH,
            torch_dtype=TORCH_TYPE,
            device_map="auto",
        ).eval()

    server_host = os.environ.get("GLM4V_HOST", "127.0.0.1")
    server_port = int(os.environ.get("GLM4V_PORT", 8000))
    if server_host not in {"127.0.0.1", "::1", "localhost"} and not API_KEY:
        raise RuntimeError("GLM4V_API_KEY is required when binding the API to a non-loopback address")
    uvicorn.run(app, host=server_host, port=server_port, workers=1)

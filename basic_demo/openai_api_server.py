import time
from asyncio.log import logger
import uvicorn
import gc
import json
import random
import string
import logging

import torch
from vllm import SamplingParams, AsyncEngineArgs, AsyncLLMEngine
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, LogitsProcessor
from sse_starlette.sse import EventSourceResponse

EventSourceResponse.DEFAULT_PING_INTERVAL = 1000

MODEL_PATH = 'THUDM/glm-4-9b-chat'
# max model length 128k
MAX_MODEL_LENGTH = 8192


@asynccontextmanager
async def lifespan(app: FastAPI):
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


def generate_id(prefix: str) -> str:
    suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=24))
    return f"{prefix}-{suffix}"


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "owner"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = ["glm-4"]


class FunctionCall(BaseModel):
    name: str
    arguments: str


class FunctionCallResponse(BaseModel):
    name: Optional[str] = None
    arguments: Optional[str] = None


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


class ChatCompletionMessageToolCall(BaseModel):
    id: Optional[str] = Field(default_factory=lambda: generate_id('call'))
    function: FunctionCall
    type: Optional[Literal["function"]] = 'function'


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system", "function", "tool"]
    content: Optional[str] = None
    function_call: Optional[FunctionCall] = None
    tool_calls: Optional[List[ChatCompletionMessageToolCall]] = None


class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "function", "system"]] = None
    content: Optional[str] = None
    function_call: Optional[FunctionCall] = None
    tool_calls: Optional[List[ChatCompletionMessageToolCall]] = None


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length", "tool_calls"]


class ChatCompletionResponseStreamChoice(BaseModel):
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length", "tool_calls"]]
    index: int


class ChatCompletionResponse(BaseModel):
    model: str
    id: str = Field(default_factory=lambda: generate_id('chatcmpl'))
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
    usage: Optional[UsageInfo] = None

    @staticmethod
    def _convert_to_tool_calls_from_content(content: str) -> Union[List[ChatCompletionMessageToolCall], str]:
        tool_calls = []
        content = content.strip()
        for response in content.split("<|assistant|>"):
            if "\n" in response:
                metadata, content = response.split("\n", maxsplit=1)
            else:
                metadata, content = "", response
            if metadata.strip():
                parameters = eval(content.strip())
                function_call = FunctionCall(
                    name=metadata.strip(),
                    arguments=json.dumps(parameters, ensure_ascii=False)
                )
                tool_calls.append(ChatCompletionMessageToolCall(function=function_call))
        return tool_calls if len(tool_calls) > 0 else content

    @staticmethod
    def stream_reply(model_id: str, content: str, finish_reason: str, use_tool: bool = False) -> str:
        if content.startswith("\n"):
            content = content[1:]
        tool_calls = None
        if use_tool:
            parsed_tool_calls = ChatCompletionResponse._convert_to_tool_calls_from_content(content)
            if isinstance(parsed_tool_calls, list):
                tool_calls = parsed_tool_calls
                finish_reason = "tool_calls"
                content = None
        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=DeltaMessage(role="assistant", content=content, tool_calls=tool_calls),
            finish_reason=finish_reason
        )
        return ChatCompletionResponse(
            model=model_id,
            choices=[choice_data],
            created=int(time.time()),
            object="chat.completion.chunk"
        ).model_dump_json(exclude_none=True)

    @staticmethod
    def reply(model_id: str, content: str, finish_reason: str, use_tool: bool = False, usage: UsageInfo = None) \
            -> 'ChatCompletionResponse':
        if content.startswith("\n"):
            content = content[1:]
        tool_calls = None
        if use_tool:
            parsed_tool_calls = ChatCompletionResponse._convert_to_tool_calls_from_content(content)
            if isinstance(parsed_tool_calls, list):
                tool_calls = parsed_tool_calls
                finish_reason = "tool_calls"
                content = None
        choice_data = ChatCompletionResponseChoice(
            index=0,
            message=ChatMessage(role="assistant", content=content, tool_calls=tool_calls),
            finish_reason=finish_reason
        )
        return ChatCompletionResponse(
            model=model_id,
            choices=[choice_data],
            created=int(time.time()),
            object="chat.completion",
            usage=usage
        )


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.8
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    tools: Optional[Union[dict, List[dict]]] = None
    tool_choice: Optional[Union[str, dict]] = None
    repetition_penalty: Optional[float] = 1.1


class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(
            self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores


@torch.inference_mode()
async def generate_stream_glm4(params):
    messages = params["messages"]
    tools = params["tools"]
    tool_choice = params["tool_choice"]
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    max_new_tokens = int(params.get("max_tokens", 8192))
    messages = process_messages(messages, tools=tools, tool_choice=tool_choice)
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    params_dict = {
        "n": 1,
        "best_of": 1,
        "presence_penalty": 1.0,
        "frequency_penalty": 0.0,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": -1,
        "repetition_penalty": repetition_penalty,
        "use_beam_search": False,
        "length_penalty": 1,
        "early_stopping": False,
        "stop_token_ids": [151329, 151336, 151338],
        "ignore_eos": False,
        "max_tokens": max_new_tokens,
        "logprobs": None,
        "prompt_logprobs": None,
        "skip_special_tokens": True,
    }
    sampling_params = SamplingParams(**params_dict)
    async for output in engine.generate(inputs=inputs, sampling_params=sampling_params, request_id=f"{time.time()}"):
        output_len = len(output.outputs[0].token_ids)
        input_len = len(output.prompt_token_ids)
        ret = {
            "text": output.outputs[0].text,
            "usage": {
                "prompt_tokens": input_len,
                "completion_tokens": output_len,
                "total_tokens": output_len + input_len
            },
            "finish_reason": output.outputs[0].finish_reason,
        }
        yield ret
    gc.collect()
    torch.cuda.empty_cache()


def process_messages(messages, tools=None, tool_choice=None):
    _messages = messages
    processed_messages = []
    msg_has_sys = False

    def filter_tools(tool_choice, tools):
        function_name = tool_choice.get('function', {}).get('name', None)
        if not function_name:
            return []
        filtered_tools = [
            tool for tool in tools
            if tool.get('function', {}).get('name') == function_name
        ]
        return filtered_tools

    if tool_choice and tool_choice != "none":
        if isinstance(tool_choice, dict):
            tools = filter_tools(tool_choice, tools)
        if tools:
            processed_messages.append(
                {
                    "role": "system",
                    "content": None,
                    "tools": tools
                }
            )
            msg_has_sys = True

    if isinstance(tool_choice, dict) and tools:
        processed_messages.append(
            {
                "role": "assistant",
                "metadata": tool_choice["function"]["name"],
                "content": ""
            }
        )

    for m in _messages:
        role, content, func_call = m.role, m.content, m.function_call
        tool_calls = getattr(m, 'tool_calls', None)

        if role == "function":
            processed_messages.append(
                {
                    "role": "observation",
                    "content": content
                }
            )
        elif role == "tool":
            processed_messages.append(
                {
                    "role": "observation",
                    "content": content,
                    "function_call": True
                }
            )
        elif role == "assistant":
            if tool_calls:
                for tool_call in tool_calls:
                    processed_messages.append(
                        {
                            "role": "assistant",
                            "metadata": tool_call.function.name,
                            "content": tool_call.function.arguments
                        }
                    )
            else:
                for response in content.split("\n"):
                    if "\n" in response:
                        metadata, sub_content = response.split("\n", maxsplit=1)
                    else:
                        metadata, sub_content = "", response
                    processed_messages.append(
                        {
                            "role": role,
                            "metadata": metadata,
                            "content": sub_content.strip()
                        }
                    )
        else:
            if role == "system" and msg_has_sys:
                msg_has_sys = False
                continue
            processed_messages.append({"role": role, "content": content})

    if not tools or tool_choice == "none":
        for m in _messages:
            if m.role == 'system':
                processed_messages.insert(0, {"role": m.role, "content": m.content})
                break
    return processed_messages


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    model_card = ModelCard(id="glm-4")
    return ModelList(data=[model_card])


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    if len(request.messages) < 1 or request.messages[-1].role == "assistant":
        raise HTTPException(status_code=400, detail="Invalid request")
    if request.tool_choice is None:
        request.tool_choice = "auto" if request.tools else "none"
    gen_params = dict(
        messages=request.messages,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens or 1024,
        echo=False,
        stream=request.stream,
        repetition_penalty=request.repetition_penalty,
        tools=request.tools,
        tool_choice=request.tool_choice,
    )
    logger.debug(f"==== request ====\n{request.model_dump_json()}")

    if request.stream:
        predict_stream_generator = predict_stream(request.model, gen_params)
        return EventSourceResponse(predict_stream_generator, media_type="text/event-stream", sep="\n")

    response = ""
    async for response in generate_stream_glm4(gen_params):
        pass
    is_tool_call = is_return_tool_call(response["text"], request.tools)
    usage = UsageInfo()
    task_usage = UsageInfo.model_validate(response["usage"])
    for usage_key, usage_value in task_usage.model_dump().items():
        setattr(usage, usage_key, getattr(usage, usage_key) + usage_value)
    return ChatCompletionResponse.reply(request.model, response["text"], response["finish_reason"], is_tool_call, usage)


def calc_max_tool_name_len(tools: Optional[List[dict]]) -> int:
    max_tool_name_len = 0
    if not tools:
        return max_tool_name_len
    tool_names = [tool['function']['name'] for tool in tools if 'function' in tool and 'name' in tool['function']]
    max_tool_name_len = max(len(tool_name) for tool_name in tool_names)
    return max_tool_name_len


def is_return_tool_call(output: str, tools: Optional[List[dict]]) -> bool:
    if not tools:
        return False
    output = output.strip()
    tool_names = [tool['function']['name'] for tool in tools if 'function' in tool and 'name' in tool['function']]
    return any(output.startswith(name) for name in tool_names)


async def predict_stream(model_id, gen_params):
    output = ""
    is_function_call = False
    has_send_first_chunk = False
    tools = gen_params.get("tools")
    max_tool_name_len = calc_max_tool_name_len(tools)
    finish_reason = "stop"

    async for new_response in generate_stream_glm4(gen_params):
        decoded_unicode = new_response["text"]
        delta_text = decoded_unicode[len(output):]
        output = decoded_unicode
        # read an extra char because the first generate char may be \n
        if len(output) <= max_tool_name_len:
            continue
        if not is_function_call:
            is_function_call = is_return_tool_call(output, tools)
        if is_function_call:
            continue
        else:
            finish_reason = new_response["finish_reason"]
            send_msg = delta_text if has_send_first_chunk else output[1:] if output.startswith("\n") else output
            has_send_first_chunk = True
            yield ChatCompletionResponse.stream_reply(model_id, send_msg, finish_reason)
    # if the total output length less than the max tool name length, has_send_first_chunk = False
    if is_function_call or not has_send_first_chunk:
        yield ChatCompletionResponse.stream_reply(model_id, output, finish_reason, is_function_call)
    yield '[DONE]'


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    engine_args = AsyncEngineArgs(
        model=MODEL_PATH,
        tokenizer=MODEL_PATH,
        tensor_parallel_size=1,
        dtype="bfloat16",
        trust_remote_code=True,
        # 占用显存的比例，请根据你的显卡显存大小设置合适的值，例如，如果你的显卡有80G，您只想使用24G，请按照24/80=0.3设置
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        worker_use_ray=False,
        engine_use_ray=False,
        disable_log_requests=True,
        max_model_len=MAX_MODEL_LENGTH,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)

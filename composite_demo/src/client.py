"""

This is a client part of composite_demo.
We provide two clients, HFClient and VLLMClient, which are used to interact with the model.
The HFClient is used to interact with the  transformers backend, and the VLLMClient is used to interact with the VLLM model.

"""

import json
from collections.abc import Generator
from copy import deepcopy
from enum import Enum, auto
from typing import Protocol

import streamlit as st

from conversation import Conversation, build_system_prompt
from tools.tool_registry import ALL_TOOLS


class ClientType(Enum):
    HF = auto()
    VLLM = auto()
    API = auto()


class Client(Protocol):
    def __init__(self, model_path: str): ...

    def generate_stream(
        self,
        tools: list[dict],
        history: list[Conversation],
        **parameters,
    ) -> Generator[tuple[str | dict, list[dict]]]: ...


def process_input(history: list[dict], tools: list[dict], role_name_replace:dict=None) -> list[dict]:
    chat_history = []
    #if len(tools) > 0:
    chat_history.append(
        {"role": "system", "content": build_system_prompt(list(ALL_TOOLS), tools)}
    )

    for conversation in history:
        role = str(conversation.role).removeprefix("<|").removesuffix("|>")
        if role_name_replace:
            role = role_name_replace.get(role, role)
        item = {
            "role": role,
            "content": conversation.content,
        }
        if conversation.metadata:
            item["metadata"] = conversation.metadata
        # Only append image for user
        if role == "user" and conversation.image:
            item["image"] = conversation.image
        chat_history.append(item)

    return chat_history


def process_response(output, history):
    content = ""
    history = deepcopy(history)
    for response in output.split("<|assistant|>"):
        if "\n" in response:
            metadata, content = response.split("\n", maxsplit=1)
        else:
            metadata, content = "", response
        if not metadata.strip():
            content = content.strip()
            history.append({"role": "assistant", "metadata": metadata, "content": content})
            content = content.replace("[[训练时间]]", "2023年")
        else:
            history.append({"role": "assistant", "metadata": metadata, "content": content})
            if history[0]["role"] == "system" and "tools" in history[0]:
                parameters = json.loads(content)
                content = {"name": metadata.strip(), "parameters": parameters}
            else:
                content = {"name": metadata.strip(), "content": content}
    return content, history


# glm-4v-9b is not available in vLLM backend, use HFClient instead.
@st.cache_resource(max_entries=1, show_spinner="Loading model...")
def get_client(model_path, typ: ClientType) -> Client:
    match typ:
        case ClientType.HF:
            from clients.hf import HFClient

            return HFClient(model_path)
        case ClientType.VLLM:
            try:
                from clients.vllm import VLLMClient
            except ImportError as e:
                e.msg += "; did you forget to install vLLM?"
                raise
            return VLLMClient(model_path)
        case ClientType.API:
            from clients.openai import APIClient
            return APIClient(model_path)

    raise NotImplementedError(f"Client type {typ} is not supported.")

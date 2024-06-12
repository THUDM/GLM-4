"""

This demo show the All tools and Long Context chat Capabilities of GLM-4.
Please follow the Readme.md to run the demo.

"""

import os
import traceback
from enum import Enum
from io import BytesIO
from uuid import uuid4

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from PIL import Image

from client import Client, ClientType, get_client
from conversation import (
    FILE_TEMPLATE,
    Conversation,
    Role,
    postprocess_text,
    response_to_str,
)
from tools.tool_registry import dispatch_tool, get_tools
from utils import extract_pdf, extract_docx, extract_pptx, extract_text


CHAT_MODEL_PATH = os.environ.get("CHAT_MODEL_PATH", "THUDM/glm-4-9b-chat")
VLM_MODEL_PATH = os.environ.get("VLM_MODEL_PATH", "THUDM/glm-4v-9b")

USE_VLLM = os.environ.get("USE_VLLM", "0") == "1"
USE_API = os.environ.get("USE_API", "0") == "1"

class Mode(str, Enum):
    ALL_TOOLS = "🛠️ All Tools"
    LONG_CTX = "📝 文档解读"
    VLM = "🖼️ 多模态"


def append_conversation(
    conversation: Conversation,
    history: list[Conversation],
    placeholder: DeltaGenerator | None = None,
) -> None:
    """
    Append a conversation piece into history, meanwhile show it in a new markdown block
    """
    history.append(conversation)
    conversation.show(placeholder)


st.set_page_config(
    page_title="GLM-4 Demo",
    page_icon=":robot:",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("GLM-4 Demo")
st.markdown(
    "<sub>智谱AI 公开在线技术文档: https://zhipu-ai.feishu.cn/wiki/RuMswanpkiRh3Ok4z5acOABBnjf </sub> \n\n <sub> 更多 GLM-4 开源模型的使用方法请参考文档。</sub>",
    unsafe_allow_html=True,
)

with st.sidebar:
    top_p = st.slider("top_p", 0.0, 1.0, 0.8, step=0.01)
    top_k = st.slider("top_k", 1, 20, 10, step=1, key="top_k")
    temperature = st.slider("temperature", 0.0, 1.5, 0.95, step=0.01)
    repetition_penalty = st.slider("repetition_penalty", 0.0, 2.0, 1.0, step=0.01)
    max_new_tokens = st.slider("max_new_tokens", 1, 4096, 2048, step=1)
    cols = st.columns(2)
    export_btn = cols[0]
    clear_history = cols[1].button("Clear", use_container_width=True)
    retry = export_btn.button("Retry", use_container_width=True)

if clear_history:
    page = st.session_state.page
    client = st.session_state.client
    st.session_state.clear()
    st.session_state.page = page
    st.session_state.client = client
    st.session_state.files_uploaded = False
    st.session_state.uploaded_texts = ""
    st.session_state.uploaded_file_nums = 0
    st.session_state.history = []

if "files_uploaded" not in st.session_state:
    st.session_state.files_uploaded = False

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid4()

if "history" not in st.session_state:
    st.session_state.history = []

first_round = len(st.session_state.history) == 0


def build_client(mode: Mode) -> Client:
    match mode:
        case Mode.ALL_TOOLS:
            st.session_state.top_k = 10
            typ = ClientType.VLLM if USE_VLLM else ClientType.HF
            typ = ClientType.API if USE_API else typ
            return get_client(CHAT_MODEL_PATH, typ)
        case Mode.LONG_CTX:
            st.session_state.top_k = 10
            typ = ClientType.VLLM if USE_VLLM else ClientType.HF
            return get_client(CHAT_MODEL_PATH, typ)
        case Mode.VLM:
            st.session_state.top_k = 1
            # vLLM is not available for VLM mode
            return get_client(VLM_MODEL_PATH, ClientType.HF)


# Callback function for page change
def page_changed() -> None:
    global client
    new_page: str = st.session_state.page
    st.session_state.history.clear()
    st.session_state.client = build_client(Mode(new_page))


page = st.radio(
    "选择功能",
    [mode.value for mode in Mode],
    key="page",
    horizontal=True,
    index=None,
    label_visibility="hidden",
    on_change=page_changed,
)

HELP = """
### 🎉 欢迎使用 GLM-4!

请在上方选取一个功能。每次切换功能时，将会重新加载模型并清空对话历史。

文档解读模式与 VLM 模式仅支持在第一轮传入文档或图像。
""".strip()

if page is None:
    st.markdown(HELP)
    exit()

if page == Mode.LONG_CTX:
    if first_round:
        uploaded_files = st.file_uploader(
            "上传文件",
            type=["pdf", "txt", "py", "docx", "pptx", "json", "cpp", "md"],
            accept_multiple_files=True,
        )
        if uploaded_files and not st.session_state.files_uploaded:
            uploaded_texts = []
            for uploaded_file in uploaded_files:
                file_name: str = uploaded_file.name
                random_file_name = str(uuid4())
                file_extension = os.path.splitext(file_name)[1]
                file_path = os.path.join("/tmp", random_file_name + file_extension)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                if file_name.endswith(".pdf"):
                    content = extract_pdf(file_path)
                elif file_name.endswith(".docx"):
                    content = extract_docx(file_path)
                elif file_name.endswith(".pptx"):
                    content = extract_pptx(file_path)
                else:
                    content = extract_text(file_path)
                uploaded_texts.append(
                    FILE_TEMPLATE.format(file_name=file_name, file_content=content)
                )
                os.remove(file_path)
            st.session_state.uploaded_texts = "\n\n".join(uploaded_texts)
            st.session_state.uploaded_file_nums = len(uploaded_files)
        else:
            st.session_state.uploaded_texts = ""
            st.session_state.uploaded_file_nums = 0
elif page == Mode.VLM:
    if first_round:
        uploaded_image = st.file_uploader(
            "上传图片",
            type=["png", "jpg", "jpeg", "bmp", "tiff", "webp"],
            accept_multiple_files=False,
        )
        if uploaded_image:
            data: bytes = uploaded_image.read()
            image = Image.open(BytesIO(data)).convert("RGB")
            st.session_state.uploaded_image = image
        else:
            st.session_state.uploaded_image = None

prompt_text = st.chat_input("Chat with GLM-4!", key="chat_input")

if prompt_text == "" and retry == False:
    print("\n== Clean ==\n")
    st.session_state.history = []
    exit()

history: list[Conversation] = st.session_state.history

if retry:
    print("\n== Retry ==\n")
    last_user_conversation_idx = None
    for idx, conversation in enumerate(history):
        if conversation.role.value == Role.USER.value:
            last_user_conversation_idx = idx
    if last_user_conversation_idx is not None:
        prompt_text = history[last_user_conversation_idx].content
        print(f"New prompt: {prompt_text}, idx = {last_user_conversation_idx}")
        del history[last_user_conversation_idx:]

for conversation in history:
    conversation.show()

tools = get_tools() if page == Mode.ALL_TOOLS else []

client: Client = st.session_state.client


def main(prompt_text: str):
    global client
    assert client is not None

    if prompt_text:
        prompt_text = prompt_text.strip()

        # Append uploaded files
        uploaded_texts = st.session_state.get("uploaded_texts")
        if page == Mode.LONG_CTX and uploaded_texts and first_round:
            meta_msg = "{} files uploaded.\n".format(
                st.session_state.uploaded_file_nums
            )
            prompt_text = uploaded_texts + "\n\n\n" + meta_msg + prompt_text
            # Clear after first use
            st.session_state.files_uploaded = True
            st.session_state.uploaded_texts = ""
            st.session_state.uploaded_file_nums = 0

        image = st.session_state.get("uploaded_image")
        if page == Mode.VLM and image and first_round:
            st.session_state.uploaded_image = None

        role = Role.USER
        append_conversation(Conversation(role, prompt_text, image=image), history)

        placeholder = st.container()
        message_placeholder = placeholder.chat_message(
            name="assistant", avatar="assistant"
        )
        markdown_placeholder = message_placeholder.empty()

        def add_new_block():
            nonlocal message_placeholder, markdown_placeholder
            message_placeholder = placeholder.chat_message(
                name="assistant", avatar="assistant"
            )
            markdown_placeholder = message_placeholder.empty()

        def commit_conversation(
            role: Role,
            text: str,
            metadata: str | None = None,
            image: str | None = None,
            new: bool = False,
        ):
            processed_text = postprocess_text(text, role.value == Role.ASSISTANT.value)
            conversation = Conversation(role, text, processed_text, metadata, image)

            # Use different placeholder for new block
            placeholder = message_placeholder if new else markdown_placeholder

            append_conversation(
                conversation,
                history,
                placeholder,
            )

        response = ""
        for _ in range(10):
            last_response = None
            history_len = None

            try:
                for response, chat_history in client.generate_stream(
                    tools=tools,
                    history=history,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    max_new_tokens=max_new_tokens,
                ):
                    if history_len is None:
                        history_len = len(chat_history)
                    elif history_len != len(chat_history):
                        commit_conversation(Role.ASSISTANT, last_response)
                        add_new_block()
                        history_len = len(chat_history)
                    last_response = response
                    replace_quote = chat_history[-1]["role"] == "assistant"
                    markdown_placeholder.markdown(
                        postprocess_text(
                            str(response) + "●", replace_quote=replace_quote
                        )
                    )
                else:
                    metadata = (
                        page == Mode.ALL_TOOLS
                        and isinstance(response, dict)
                        and response.get("name")
                        or None
                    )
                    role = Role.TOOL if metadata else Role.ASSISTANT
                    text = (
                        response.get("content")
                        if metadata
                        else response_to_str(response)
                    )
                    commit_conversation(role, text, metadata)
                    if metadata:
                        add_new_block()
                        try:
                            with markdown_placeholder:
                                with st.spinner(f"Calling tool {metadata}..."):
                                    observations = dispatch_tool(
                                        metadata, text, str(st.session_state.session_id)
                                    )
                        except Exception as e:
                            traceback.print_exc()
                            st.error(f'Uncaught exception in `"{metadata}"`: {e}')
                            break

                        for observation in observations:
                            observation.text = observation.text
                            commit_conversation(
                                Role.OBSERVATION,
                                observation.text,
                                observation.role_metadata,
                                observation.image_url,
                                new=True,
                            )
                            add_new_block()
                        continue
                    else:
                        break
            except Exception as e:
                traceback.print_exc()
                st.error(f"Uncaught exception: {traceback.format_exc()}")
        else:
            st.error("Too many chaining function calls!")


main(prompt_text)

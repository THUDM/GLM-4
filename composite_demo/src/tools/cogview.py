import streamlit as st
from zhipuai import ZhipuAI
from zhipuai.types.image import GeneratedImage

from .config import COGVIEW_MODEL, ZHIPU_AI_KEY
from .interface import ToolObservation

@st.cache_resource
def get_zhipu_client():
    return ZhipuAI(api_key=ZHIPU_AI_KEY)

def map_response(img: GeneratedImage):
    return ToolObservation(
        content_type='image',
        text='CogView 已经生成并向用户展示了生成的图片。',
        image_url=img.url,
        role_metadata='cogview_result'
    )

def tool_call(prompt: str, session_id: str) -> list[ToolObservation]:
    client = get_zhipu_client()
    response = client.images.generations(model=COGVIEW_MODEL, prompt=prompt).data
    return list(map(map_response, response))

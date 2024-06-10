import os
import torch
from threading import Thread
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from transformers import (
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer, AutoModel
)
from pydantic import BaseModel
from PIL import Image
import base64
import io

app = FastAPI()

MODEL_PATH = os.environ.get('MODEL_PATH', '/root/.cache/modelscope/hub/ZhipuAI/glm-4v-9b')

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

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    temperature: float = 0.6
    top_p: float = 0.8
    max_tokens: int = 1024
    image: str = None

@app.post("/v1/chat/completions")
async def chat(chat_request: ChatRequest):
    messages = chat_request.messages
    temperature = chat_request.temperature
    top_p = chat_request.top_p
    max_length = chat_request.max_tokens
    image_data = chat_request.image

    inputs = []
    for message in messages:
        inputs.append({"role": message.role, "content": message.content})

    if image_data != "-1":
        try:
            image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert("RGB")
        except:
            raise HTTPException(status_code=400, detail="Invalid image data")
        inputs[-1].update({"image": image})

    model_inputs = tokenizer.apply_chat_template(
        inputs,
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

    stop = StopOnTokens()
    generate_kwargs = {
        **model_inputs,
        "streamer": streamer,
        "max_new_tokens": max_length,
        "do_sample": True,
        "top_p": top_p,
        "temperature": temperature,
        "stopping_criteria": StoppingCriteriaList([stop]),
        "repetition_penalty": 1.2,
        "eos_token_id": [151329, 151336, 151338],
    }

    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    response = ""
    for new_token in streamer:
        if new_token:
            response += new_token

    return JSONResponse(content={"choices": [{"message": {"role": "assistant", "content": response.strip()}}]})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

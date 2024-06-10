import requests
import base64

def chat_with_model(messages, image_path=None):
    url = "http://localhost:8000/v1/chat/completions"
    image_data = "-1"
    if image_path:
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")

    payload = {
        "model": "glm-4v-9b",
        "messages": messages,
        "temperature": 0.6,
        "top_p": 0.8,
        "max_tokens": 1024,
        "image": image_data
    }

    response = requests.post(url, json=payload)
    return response.json()

# Example usage
messages = [
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "user", "content": "提取文本原样输出,无需解释"}
]

# Chat with model without image
response = chat_with_model(messages)
print(response)

# Chat with model with image
response_with_image = chat_with_model(messages, image_path="/home/image001.png")
print(response_with_image)

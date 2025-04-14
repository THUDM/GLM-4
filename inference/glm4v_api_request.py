"""
This script creates a OpenAI Request demo for the glm-4v-9b model, just Use OpenAI API to interact with the model.
For LLM such as GLM-4-9B-0414, using with vLLM OpenAI Server.

vllm serve THUDM/GLM-4-32B-Chat-0414 --tensor_parallel_size 4

"""

import base64

from openai import OpenAI


base_url = "http://127.0.0.1:8000/v1/"
client = OpenAI(api_key="EMPTY", base_url=base_url)


def create_chat_completion(messages, use_stream=False):
    response = client.chat.completions.create(
        model="glm-4v",
        messages=messages,
        stream=use_stream,
        max_tokens=256,
        temperature=0.4,
        presence_penalty=1.2,
        top_p=0.8,
    )
    if response:
        if use_stream:
            for chunk in response:
                print(chunk)
        else:
            print(response)
    else:
        print("Error:", response.status_code)


def encode_image(image_path):
    """
    Encodes an image file into a base64 string.
    Args:
        image_path (str): The path to the image file.

    This function opens the specified image file, reads its content, and encodes it into a base64 string.
    The base64 encoding is used to send images over HTTP as text.
    """

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def glm4v_simple_image_chat(use_stream=False, img_path=None):
    """
    Facilitates a simple chat interaction involving an image.

    Args:
        use_stream (bool): Specifies whether to use streaming for chat responses.
        img_path (str): Path to the image file to be included in the chat.

    This function encodes the specified image and constructs a predefined conversation involving the image.
    It then calls `create_chat_completion` to generate a response from the model.
    The conversation includes asking about the content of the image and a follow-up question.
    """

    img_url = f"data:image/jpeg;base64,{encode_image(img_path)}"
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What’s in this image?",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": img_url},
                },
            ],
        },
        {
            "role": "assistant",
            "content": "The image displays a wooden boardwalk extending through a vibrant green grassy wetland. The sky is partly cloudy with soft, wispy clouds, indicating nice weather. Vegetation is seen on either side of the boardwalk, and trees are present in the background, suggesting that this area might be a natural reserve or park designed for ecological preservation and outdoor recreation. The boardwalk allows visitors to explore the area without disturbing the natural habitat.",
        },
        {"role": "user", "content": "Do you think this is a spring or winter photo?"},
    ]
    create_chat_completion(messages=messages, use_stream=use_stream)


if __name__ == "__main__":
    glm4v_simple_image_chat(use_stream=False, img_path="demo.jpg")

"""
This script creates a OpenAI Request demo for the glm-4-9b model, just Use OpenAI API to interact with the model.
"""

from openai import OpenAI

base_url = "http://127.0.0.1:8000/v1/"
client = OpenAI(api_key="EMPTY", base_url=base_url)


def function_chat(use_stream=False):
    messages = [{"role": "user", "content": "What's the Celsius temperature in San Francisco?"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "format": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The temperature unit to use. Infer this from the users location.",
                        },
                    },
                    "required": ["location", "format"],
                },
            }
        },
    ]

    # # All Tools 能力: 绘图
    # messages = [{"role": "user", "content": "帮我画一张天空的画画吧"}]
    # tools = [{"type": "cogview"}]
    #
    # All Tools 能力: 联网查询
    # messages = [{"role": "user", "content": "今天黄金的价格"}]
    # tools = [{"type": "simple_browser"}]

    response = client.chat.completions.create(
        model="glm-4",
        messages=messages,
        tools=tools,
        stream=use_stream,
        max_tokens=256,
        temperature=0.9,
        presence_penalty=1.2,
        top_p=0.1,
        tool_choice="auto",  # use "auto" to let the model choose the tool automatically
        # tool_choice={"type": "function", "function": {"name": "my_function"}},
    )
    if response:
        if use_stream:
            for chunk in response:
                print(chunk)
        else:
            print(response)
    else:
        print("Error:", response.status_code)


def simple_chat(use_stream=False):
    messages = [
        {
            "role": "system",
            "content": "请在你输出的时候都带上“喵喵喵”三个字，放在开头。",
        },
        {
            "role": "user",
            "content": "你好，你是谁"
        }
    ]
    response = client.chat.completions.create(
        model="glm-4",
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


if __name__ == "__main__":
    simple_chat(use_stream=True)
    # function_chat(use_stream=False) # Only False is supported

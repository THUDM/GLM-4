"""
OpenAI API client.
"""
from openai import OpenAI
from collections.abc import Generator

from client import Client, process_input, process_response
from conversation import Conversation

def format_openai_tool(origin_tools):
    openai_tools = []
    for tool in origin_tools:
        openai_param={}
        for param in tool['params']:
            openai_param[param['name']] = {}
        openai_tool = {
            "type": "function",
            "function": {
                "name": tool['name'],
                "description": tool['description'],
                "parameters": {
                    "type": "object",
                    "properties": {
                        param['name']:{'type':param['type'], 'description':param['description']} for param in tool['params']
                    },
                    "required": [param['name'] for param in tool['params'] if param['required']]
                    }
                }
            }
        openai_tools.append(openai_tool)
    return openai_tools

class APIClient(Client):
    def __init__(self, model_path: str):
        base_url = "http://127.0.0.1:8000/v1/"
        self.client = OpenAI(api_key="EMPTY", base_url=base_url)
        self.use_stream= False
        self.role_name_replace = {'observation':'tool'}

    def generate_stream(
        self,
        tools: list[dict],
        history: list[Conversation],
        **parameters,
    ) -> Generator[tuple[str | dict, list[dict]]]:
        chat_history = process_input(history, '', role_name_replace=self.role_name_replace)
        #messages = process_input(history, '', role_name_replace=self.role_name_replace)
        openai_tools = format_openai_tool(tools)
        response = self.client.chat.completions.create(
            model="glm-4",
            messages=chat_history,
            tools=openai_tools,
            stream=self.use_stream,
            max_tokens=parameters["max_new_tokens"],
            temperature=parameters["temperature"],
            presence_penalty=1.2,
            top_p=parameters["top_p"],
            tool_choice="auto"
        )
        output = response.choices[0].message
        if output.tool_calls:
            glm4_output = output.tool_calls[0].function.name + '\n' + output.tool_calls[0].function.arguments
        else:
            glm4_output = output.content
        yield process_response(glm4_output, chat_history)

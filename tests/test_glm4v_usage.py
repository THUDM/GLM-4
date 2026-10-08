from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from inference import glm4v_server as server


class ModelInputs(dict):
    def to(self, device):
        return self


class Tokenizer:
    def apply_chat_template(self, *args, **kwargs):
        return ModelInputs(input_ids=torch.tensor([[1, 2, 3, 4, 5]]))

    def encode(self, text):
        return text.split()


@pytest.mark.parametrize("pieces", [[], ["one"], ["one ", "two"]])
def test_usage_counts_prompt_and_completion_separately(pieces):
    model = SimpleNamespace(parameters=lambda: iter([torch.zeros(1)]),
                            generate=lambda **kwargs: None)
    with patch.object(server, "process_history_and_images", return_value=("hi", [], [])), \
            patch.object(server, "TextIteratorStreamer", return_value=iter(pieces)):
        results = list(server.generate_stream_glm4v(model, Tokenizer(), {"messages": []}))
    for result in results:
        completion = len(result["text"].split())
        assert result["usage"] == {
            "prompt_tokens": 5,
            "completion_tokens": completion,
            "total_tokens": 5 + completion,
        }

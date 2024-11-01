import os
import yaml
import dashscope
from http import HTTPStatus
from typing import List, Dict, Any
from mmte.utils.registry import registry
from mmte.models.base import BaseChat, Response
from mmte.utils.utils import get_abs_path


def transform_messages(messages):
    transformed_messages = []
    for message in messages:
        role = message.get("role")
        if isinstance(message.get("content"), dict):
            content_dict = message.get("content")
            transformed_content = []
            if "text" in content_dict:
                transformed_content.append({"text": content_dict["text"]})
            if "image_path" in content_dict:
                transformed_content.append({"image": content_dict["image_path"]})
        else:
            transformed_content = []
            transformed_content.append({"text": message.get("content")})
        transformed_messages.append({"role": role, "content": transformed_content})

    return transformed_messages


@registry.register_chatmodel()
class QwenPlusChat(BaseChat):
    """
    Chat class for Qwen-Plus API, handling multimodal generation.
    """

    MODEL_CONFIG = {"qwen-vl-plus": "configs/models/qwen/qwen.yaml"}

    model_family = list(MODEL_CONFIG.keys())

    model_arch = "qwen-plus"

    def __init__(self, model_id: str = "qwen-vl-plus", **kargs):
        super().__init__(model_id=model_id)
        config = self.MODEL_CONFIG[self.model_id]
        with open(get_abs_path(config)) as f:
            self.model_config = yaml.load(f, Loader=yaml.FullLoader)
        api_key = os.getenv("qwen_apikey", "")
        assert api_key, "qwen_apikey is empty"
        self.api_key = api_key

    def chat(self, messages: List[Dict[str, Any]], **generation_kwargs):
        """Simple single round multimodal conversation call."""
        generation_config = {
            "max_length": generation_kwargs.get("max_new_tokens", 512),
            "top_k": None if generation_kwargs.get("do_sample", False) else 1,
            "seed": 1234,
        }
        from pprint import pp

        pp(generation_config)
        messages = transform_messages(messages)
        response = dashscope.MultiModalConversation.call(
            model="qwen-vl-plus",
            messages=messages,
            **generation_config,
        )
        # The response status_code is HTTPStatus.OK indicate success,
        # otherwise indicate request is failed, you can get error code
        # and message from code and message.
        if response.status_code == HTTPStatus.OK:
            if "text" in response.output.choices[0].message.content[0].keys():
                response_message = response.output.choices[0].message.content[0]["text"]
            elif "box" in response.output.choices[0].message.content[0].keys():
                response_message = response.output.choices[0].message.content[0]["box"]
            finish_reason = response.output.choices[0].finish_reason
            logprobs = None
            return Response(self.model_id, response_message, logprobs, finish_reason)
        else:
            print(response.code)  # The error code.
            print(response.message)  # The error message.
            return Response(
                self.model_id, f"Error in generation:{response.message}", None, None
            )

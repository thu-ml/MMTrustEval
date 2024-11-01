from typing import List, Dict, Any, Literal
import anthropic
import yaml
from mmte.utils.registry import registry
from mmte.models.base import BaseChat, Response
from mmte.utils.utils import get_abs_path
import os
import base64
import time
import io
import httpx
from PIL import Image
import imghdr


def encode_image(image_path: str):
    buffer = io.BytesIO()

    img = Image.open(image_path).convert("RGB")
    print(img.size)
    if img.width > 400 or img.height > 400:
        if img.width > img.height:
            new_width = 400
            concat = float(new_width / float(img.width))
            new_height = int((float(img.height) * float(concat)))
        else:
            new_height = 400
            concat = float(new_height / float(img.height))
            new_width = int((float(img.width) * float(concat)))
        img = img.resize((new_width, new_height), Image.LANCZOS)

    format = imghdr.what(image_path)
    if not format:
        format = "JPEG"
    img.save(buffer, format=format)
    img_data = base64.b64encode(buffer.getvalue())
    return img_data.decode("utf-8")


@registry.register_chatmodel()
class ClaudeChat(BaseChat):
    """
    Chat class for Anthropic models, e.g., claude-3
    """

    MODEL_CONFIG = {
        "claude-3-sonnet-20240229": "configs/models/anthropic/anthropic.yaml",
        "claude-3-5-sonnet-20240620": "configs/models/anthropic/anthropic.yaml",
    }

    model_family = list(MODEL_CONFIG.keys())

    model_arch = "claude"

    def __init__(self, model_id: str = "claude-3-sonnet-20240229", **kargs):
        super().__init__(model_id=model_id)
        config = self.MODEL_CONFIG[self.model_id]
        with open(get_abs_path(config)) as f:
            self.model_config = yaml.load(f, Loader=yaml.FullLoader)

        api_key = os.getenv("anthropic_apikey", "")
        assert api_key, "anthropic_apikey is empty"
        self.api_key = api_key
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.max_retries = self.model_config.get("max_retries", 10)
        self.timeout = self.model_config.get("timeout", 1)

    def chat(self, messages: List[Dict[str, Any]], **generation_kwargs):
        conversation = []
        for message in messages:
            if message["role"] in ["system", "user", "assistant"]:
                if isinstance(message["content"], dict):
                    # multimodal content
                    text = message["content"]["text"]
                    image_data = encode_image(message["content"]["image_path"])
                    format = imghdr.what(message["content"]["image_path"])
                    if not format:
                        format = "jpeg"
                    image_media_type = "image/{}".format(format)
                    content = [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": image_media_type,
                                "data": image_data,
                            },
                        },
                        {"type": "text", "text": text},
                    ]
                else:
                    text = message["content"]
                    content = [{"type": "text", "text": text}]
                conversation.append({"role": message["role"], "content": content})
            else:
                raise ValueError(
                    "Unsupported role. Only system, user and assistant are supported."
                )

        # Create Completion Request
        raw_request: Dict[str, Any] = {
            "model": self.model_id,
            "messages": conversation,
            "max_tokens": generation_kwargs.get("max_new_tokens"),
            "temperature": 1 if generation_kwargs.get("do_sample", False) else 0,
        }

        from pprint import pp
        pp(raw_request)
        for i in range(self.max_retries):
            try:
                response = self.client.messages.create(**raw_request)
                break
            except Exception as e:
                print(f"Error in generation: {e}")
                response = f"Error in generation: {e}"
                time.sleep(self.timeout)
        if isinstance(response, str):
            return Response(self.model_id, response, None, None)
        response_message = response.content[0].text
        finish_reason = response.stop_reason
        logprobs = None

        return Response(self.model_id, response_message, logprobs, finish_reason)

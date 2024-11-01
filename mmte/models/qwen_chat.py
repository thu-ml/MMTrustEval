from typing import List
import torch
from mmte.models.base import BaseChat, Response
from transformers import AutoModelForCausalLM, AutoTokenizer
from mmte.utils.registry import registry


@registry.register_chatmodel()
class QwenChat(BaseChat):
    """
    Chat class for Qwen models
    """

    MODEL_CONFIG = {
        "qwen-vl-chat": "Qwen/Qwen-VL-Chat",
    }

    model_family = list(MODEL_CONFIG.keys())

    model_arch = "qwen"

    def __init__(self, model_id: str, device: str = "cuda:0"):
        super().__init__(model_id)
        config = self.MODEL_CONFIG[self.model_id]
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(config, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            config, device_map=self.device, trust_remote_code=True
        ).eval()

    @torch.no_grad()
    def chat(self, messages: List, **generation_kwargs):
        assert len(messages) == 1, "Only support one-turn conversation currently"
        for message in messages:
            if message["role"] in ["system", "user", "assistant"]:
                if message["role"] == "user":
                    if isinstance(message["content"], dict):
                        # multimodal
                        image_path = message["content"]["image_path"]
                        user_message = message["content"]["text"]
                        query = self.tokenizer.from_list_format(
                            [
                                {"image": image_path},
                                {"text": user_message},
                            ]
                        )
                    else:
                        # text only conversation
                        user_message = message["content"]
                        query = self.tokenizer.from_list_format(
                            [
                                {"text": user_message},
                            ]
                        )

                elif message["role"] == "assistant":
                    # TODO: add assistant answer into the conversation
                    pass
            else:
                raise ValueError(
                    "Unsupported role. Only system, user and assistant are supported."
                )
        from pprint import pp

        pp(generation_kwargs)
        response, history = self.model.chat(
            self.tokenizer, query=query, history=None, **generation_kwargs
        )

        return Response(self.model_id, response, None, None)

from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import List
from mmte.models.base import BaseChat, Response
from mmte.utils.registry import registry
import torch

torch.set_grad_enabled(False)


# init model and tokenizer
@registry.register_chatmodel()
class InternLMXComposerChat(BaseChat):
    """
    Chat class for InternLM-XComposer models
    """

    # TODO: update model config
    MODEL_CONFIG = {
        "internlm-xcomposer-7b": "internlm/internlm-xcomposer-7b",
        "internlm-xcomposer2-vl-7b": "internlm/internlm-xcomposer2-vl-7b",
    }

    model_family = list(MODEL_CONFIG.keys())

    model_arch = "internlm-xcomposer"

    def __init__(self, model_id: str, device: str = "cuda:0"):
        super().__init__(model_id)
        self.device = device
        # self.model = AutoModel.from_pretrained(self.MODEL_CONFIG[model_id], trust_remote_code=True).to(self.device).eval()
        config = AutoConfig.from_pretrained(
            self.MODEL_CONFIG[model_id], trust_remote_code=True
        )
        config.device = self.device
        self.model = (
            AutoModel.from_pretrained(
                self.MODEL_CONFIG[model_id], config=config, trust_remote_code=True
            )
            .to(self.device)
            .eval()
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_CONFIG[model_id], trust_remote_code=True
        )
        self.model.tokenizer = self.tokenizer
        self.model_id = model_id

    @torch.no_grad()
    def chat(self, messages: List, **generation_kwargs):
        assert len(messages) == 1, "Only support one-turn conversation currently"
        generation_config = {
            "do_sample": False,
            "max_new_tokens": 128,
        }
        generation_config.update(generation_kwargs)
        from pprint import pp

        pp(generation_config)

        for message in messages:
            if message["role"] in ["system", "user", "assistant"]:
                if message["role"] == "user":
                    if isinstance(message["content"], dict):
                        # multimodal
                        image_path = message["content"]["image_path"]
                        user_message = message["content"]["text"]
                        if self.model_id == "internlm-xcomposer2-vl-7b":
                            user_message = "<ImageHere>" + user_message
                            response, _ = self.model.chat(
                                self.tokenizer,
                                query=user_message,
                                image=image_path,
                                **generation_config,
                            )
                        else:
                            response = self.model.generate(
                                user_message,
                                image_path,
                                **generation_config,
                            )
                    else:
                        user_message = message["content"]
                        if self.model_id == "internlm-xcomposer2-vl-7b":
                            response, _ = self.model.chat(
                                self.tokenizer,
                                query=user_message,
                                **generation_config,
                            )
                        else:
                            response = self.model.generate(
                                user_message,
                                **generation_config,
                            )
                elif message["role"] == "assistant":
                    # TODO: add assistant answer into the conversation
                    pass
            else:
                raise ValueError(
                    "Unsupported role. Only system, user and assistant are supported."
                )

        scores = None

        return Response(self.model_id, response, scores, None)

from typing import List

import torch
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from mmte.models.base import BaseChat, Response
from mmte.utils.registry import registry
from mmte.utils.utils import get_abs_path


@registry.register_chatmodel()
class mPLUGOwl3Chat(BaseChat):
    """
    Chat class for mPLUG-Owl3 models
    """

    MODEL_CONFIG = {
        "mplug-owl3-7b-240728": "configs/models/mplug-owl3/mplug-owl3-7b-240728.yaml",
    }

    model_family = list(MODEL_CONFIG.keys())

    model_arch = "mplug-owl3"

    def __init__(self, model_id: str, device: str = "cuda:0"):
        super().__init__(model_id)
        config = self.MODEL_CONFIG[self.model_id]
        self.config = OmegaConf.load(get_abs_path(config))

        model_path = self.config.model.model_path
        model = (
            AutoModel.from_pretrained(
                model_path,
                attn_implementation="sdpa",
                torch_dtype=torch.half,
                trust_remote_code=True,
            )
            .eval()
            .to(device)
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        processor = model.init_processor(tokenizer)

        self.device = device
        self.tokenizer, self.model, self.processor = tokenizer, model, processor

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
                        images = [Image.open(image_path).convert("RGB")]
                        messages = [
                            {
                                "role": "user",
                                "content": """<|image|>{}.""".format(user_message),
                            },
                            {"role": "assistant", "content": ""},
                        ]

                    else:
                        user_message = message["content"]
                        images = None
                        messages = [
                            {
                                "role": "user",
                                "content": """{}.""".format(user_message),
                            },
                            {"role": "assistant", "content": ""},
                        ]

                elif message["role"] == "assistant":
                    # TODO: add assistant answer into the conversation
                    pass
            else:
                raise ValueError(
                    "Unsupported role. Only system, user and assistant are supported."
                )

        inputs = self.processor(messages, images=images, videos=None)

        generation_config = {
            "do_sample": False,
            "max_new_tokens": self.config.parameters.max_new_tokens,
            "temperature": self.config.parameters.temperature,
        }
        generation_config.update(generation_kwargs)

        from pprint import pp

        pp(generation_config)
        max_new_tokens = generation_config["max_new_tokens"]
        temperature = generation_config["temperature"]
        do_sample = generation_kwargs["do_sample"]

        inputs.to("cuda")
        inputs.update(
            {
                "tokenizer": self.tokenizer,
                "max_new_tokens": max_new_tokens,
                "decode_text": True,
            }
        )

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                do_sample=do_sample,
                temperature=temperature,
            )[0]

        scores = None

        return Response(self.model_id, outputs, scores, None)

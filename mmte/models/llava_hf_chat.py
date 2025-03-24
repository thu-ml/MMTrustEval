from typing import List

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

from mmte.models.base import BaseChat, Response
from mmte.utils.registry import registry


@registry.register_chatmodel()
class LLaVAHFChat(BaseChat):

    # TODO: update model config
    MODEL_CONFIG = {
        "llava-1.5-7b-hf": "llava-hf/llava-1.5-7b-hf",
        "llava-1.5-13b-hf": "llava-hf/llava-1.5-13b-hf",
    }
    model_family = list(MODEL_CONFIG.keys())

    def __init__(self, model_id: str, device: str = "cuda:0"):
        super().__init__(model_id)
        self.device = device
        model_id = self.MODEL_CONFIG[self.model_id]

        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(0)

        self.processor = AutoProcessor.from_pretrained(model_id)

    @torch.no_grad()
    def chat(self, messages: List, **generation_kwargs):
        # TODO: if system message provided.
        assert len(messages) == 1, "Only support one-turn conversation currently"
        for message in messages:
            if message["role"] in ["system", "user", "assistant"]:
                if message["role"] == "user":
                    if isinstance(message["content"], dict):
                        # multimodal
                        image_path = message["content"]["image_path"]
                        user_message = message["content"]["text"]
                        conversation = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": user_message},
                                    {"type": "image"},
                                ],
                            },
                        ]
                        raw_image = Image.open(image_path)
                    else:
                        user_message = message["content"]
                        conversation = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": user_message},
                                ],
                            },
                        ]
                        raw_image = None
                elif message["role"] == "assistant":
                    # TODO: add assistant answer into the conversation
                    pass
            else:
                raise ValueError(
                    "Unsupported role. Only system, user and assistant are supported."
                )

        generation_config = {
            "max_new_tokens": 200,
            "do_sample": False,
        }
        generation_config.update(generation_kwargs)

        prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        inputs = self.processor(images=raw_image, text=prompt, return_tensors="pt").to(
            0, torch.float16
        )
        from pprint import pp

        pp(generation_config)
        output = self.model.generate(**inputs, **generation_config)
        output = self.processor.decode(
            output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()

        scores = None

        return Response(self.model_id, output, scores, None)

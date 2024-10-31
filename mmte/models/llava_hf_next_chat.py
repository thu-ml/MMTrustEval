from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from typing import List
from PIL import Image
import torch
from mmte.models.base import BaseChat, Response
from mmte.utils.registry import registry


@registry.register_chatmodel()
class LLaVANEXTChat(BaseChat):

    # TODO: update model config
    MODEL_CONFIG = {
        "llava-v1.6-mistral-7b-hf": "llava-hf/llava-v1.6-mistral-7b-hf",
        "llava-v1.6-vicuna-7b-hf": "llava-hf/llava-v1.6-vicuna-7b-hf",
        "llava-v1.6-vicuna-13b-hf": "llava-hf/llava-v1.6-vicuna-13b-hf",
        "llama3-llava-next-8b-hf": "llava-hf/llama3-llava-next-8b-hf",
    }
    model_family = list(MODEL_CONFIG.keys())

    def __init__(self, model_id: str, device: str = "cuda:0"):
        super().__init__(model_id)
        self.device = device
        model_id = self.MODEL_CONFIG[self.model_id]

        self.processor = LlavaNextProcessor.from_pretrained(model_id)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True
        )
        self.model.to(device)

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

        default_generation_config = {
            "max_new_tokens": 100,
            "do_sample": False,
        }
        default_generation_config.update(generation_kwargs)

        prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        inputs = self.processor(images=raw_image, text=prompt, return_tensors="pt").to(
            0, torch.float16
        )

        output = self.model.generate(**inputs, **default_generation_config)
        output = self.processor.decode(
            output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()

        scores = None

        return Response(self.model_id, output, scores, None)

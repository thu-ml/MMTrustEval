from typing import List

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

from mmte.models.base import BaseChat, Response
from mmte.utils.registry import registry


@registry.register_chatmodel()
class MolmoChat(BaseChat):
    """
    Chat class for allenai/Molmo-7B-D-0924 model
    """

    MODEL_CONFIG = {"molmo-7b-d-0924": "allenai/Molmo-7B-D-0924"}

    model_family = list(MODEL_CONFIG.keys())

    model_arch = "molmo"

    def __init__(self, model_id: str, device: str = "cuda:0"):
        super().__init__(model_id)
        model_path = self.MODEL_CONFIG[self.model_id]
        self.device = device

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto",
        )

        # load the model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto",
        )

    @torch.no_grad()
    def chat(self, messages: List, **generation_kwargs):
        assert len(messages) == 1, "Only support one-turn conversation currently"
        for message in messages:
            if message["role"] in ["system", "user", "assistant"]:
                if message["role"] == "user":
                    if isinstance(message["content"], dict):
                        # multimodal
                        image_path = message["content"]["image_path"]
                        text = message["content"]["text"]

                        inputs = self.processor.process(
                            images=[Image.open(image_path)], text=text
                        )
                    else:
                        # text only conversation
                        text = message["content"]
                        inputs = self.processor.process(images=None, text=text)
                elif message["role"] == "assistant":
                    # TODO: add assistant answer into the conversation
                    pass
            else:
                raise ValueError(
                    "Unsupported role. Only system, user and assistant are supported."
                )
        inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}

        generation_config = {
            "max_new_tokens": 200,
            "do_sample": False,
            "stop_strings": "<|endoftext|>",
        }
        generation_config.update(generation_kwargs)

        from pprint import pp

        pp(generation_config)

        output = self.model.generate_from_batch(
            inputs,
            GenerationConfig(**generation_config),
            tokenizer=self.processor.tokenizer,
        )

        generated_tokens = output[0, inputs["input_ids"].size(1) :]
        generated_text = self.processor.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        ).strip()

        return Response(self.model_id, generated_text, None, None)

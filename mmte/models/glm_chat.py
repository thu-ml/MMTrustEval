from pprint import pp
from typing import List

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

from mmte.models.base import BaseChat, Response
from mmte.utils.registry import registry


@registry.register_chatmodel()
class GLMChat(BaseChat):
    """
    Chat class for THUDM/glm-4v-9b
    """

    MODEL_CONFIG = {"glm-4v-chat": "THUDM/glm-4v-9b"}

    model_family = list(MODEL_CONFIG.keys())

    model_arch = "glm"

    def __init__(self, model_id: str, device: str = "cuda:0"):
        super().__init__(model_id)
        model_name = self.MODEL_CONFIG[self.model_id]
        self.device = device
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            .to(device)
            .eval()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
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
                        image = Image.open(image_path).convert("RGB")

                        inputs = self.tokenizer.apply_chat_template(
                            [{"role": "user", "image": image, "content": text}],
                            add_generation_prompt=True,
                            tokenize=True,
                            return_tensors="pt",
                            return_dict=True,
                        )

                    else:
                        # text only conversation
                        text = message["content"]
                        inputs = self.tokenizer.apply_chat_template(
                            [{"role": "user", "content": text}],
                            add_generation_prompt=True,
                            tokenize=True,
                            return_tensors="pt",
                            return_dict=True,
                        )

                elif message["role"] == "assistant":
                    # TODO: add assistant answer into the conversation
                    pass
            else:
                raise ValueError(
                    "Unsupported role. Only system, user and assistant are supported."
                )

        generation_config = dict(max_new_tokens=512, do_sample=False)
        generation_config.update(generation_kwargs)
        pp(generation_config)

        with torch.no_grad():
            inputs = inputs.to(self.model.device)
            outputs = self.model.generate(**inputs, **generation_config)
            outputs = outputs[:, inputs["input_ids"].shape[1] :]
            outputs = self.tokenizer.decode(outputs[0])

        response = outputs.rstrip("<|endoftext|>").strip()
        return Response(self.model_id, response, None, None)

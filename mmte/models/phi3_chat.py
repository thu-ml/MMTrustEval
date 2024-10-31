from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import transformers
from transformers import AutoModelForCausalLM
from transformers import AutoProcessor
from PIL import Image
import torch
import requests
from mmte.utils.registry import registry
import sys
import os
from mmte.models.base import BaseChat, Response


@registry.register_chatmodel()
class Phi3Chat(BaseChat):

    model_family = ["phi-3"]
    model_arch = "phi-3"

    def __init__(self, model_id: str, device: str = "cuda:0", bf16: bool = True):
        super().__init__(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-vision-128k-instruct",
            device_map=device,
            trust_remote_code=True,
            torch_dtype="auto",
            _attn_implementation="eager",
        )  # use _attn_implementation='eager' to disable flash attention
        self.processor = AutoProcessor.from_pretrained(
            "microsoft/Phi-3-vision-128k-instruct", trust_remote_code=True
        )

    @torch.no_grad()
    def chat(self, messages: List, **generation_kwargs):

        image = None

        for m in messages:
            if isinstance(m["content"], dict):
                image = [Image.open(m["content"]["image_path"])]
                m["content"] = "<|image_1|>\n" + m["content"]["text"]

        prompt = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(prompt, image, return_tensors="pt").to("cuda:0")

        generation_kwargs["return_dict_in_generate"] = True
        default_generation_config = {
            "max_new_tokens": 500,
            "temperature": 0.0,
            "do_sample": False,
        }
        default_generation_config.update(generation_kwargs)

        outputs = self.model.generate(
            **inputs,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            **default_generation_config
        )
        generate_ids = outputs.sequences
        # remove input tokens
        generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
        response = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        scores = None
        if "scores" in outputs.keys() and outputs.scores[0].shape[0] == 1:
            scores = torch.cat(outputs.scores).cpu().numpy()

        return Response(self.model_id, response, scores, None)

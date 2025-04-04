from typing import List

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from mmte.models.base import BaseChat, Response
from mmte.utils.registry import registry


@registry.register_chatmodel()
class Phi3Chat(BaseChat):

    MODEL_CONFIG = {
        "phi-3": "microsoft/Phi-3-vision-128k-instruct",
        "phi-3.5": "microsoft/Phi-3.5-vision-instruct",
    }
    model_family = list(MODEL_CONFIG.keys())
    model_arch = "phi"

    def __init__(self, model_id: str, device: str = "cuda:0", bf16: bool = True):
        super().__init__(model_id)
        model_path = self.MODEL_CONFIG[self.model_id]
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            trust_remote_code=True,
            torch_dtype="auto",
            _attn_implementation="eager",
        )  # use _attn_implementation='eager' to disable flash attention
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )

    @torch.no_grad()
    def chat(self, messages: List, **generation_kwargs):

        image = None
        assert len(messages) == 1, "Only support one-turn conversation currently"
        for m in messages:
            if isinstance(m["content"], dict):
                image = [Image.open(m["content"]["image_path"])]
                m["content"] = "<|image_1|>\n" + m["content"]["text"]

        prompt = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(prompt, image, return_tensors="pt").to("cuda:0")

        generation_kwargs["return_dict_in_generate"] = True
        generation_config = {
            "max_new_tokens": 500,
            "temperature": 0.0,
            "do_sample": False,
        }
        generation_config.update(generation_kwargs)

        from pprint import pp

        pp(generation_config)

        outputs = self.model.generate(
            **inputs,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            **generation_config
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

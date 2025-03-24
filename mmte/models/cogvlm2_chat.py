from typing import List

import torch
import yaml
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

from mmte.models.base import BaseChat, Response
from mmte.utils.registry import registry
from mmte.utils.utils import get_abs_path


@registry.register_chatmodel()
class CogVLM2Chat(BaseChat):
    """
    Chat class for CogVLM2 models
    """

    MODEL_CONFIG = {
        "cogvlm2-llama3-chat-19b": "configs/models/cogvlm/cogvlm2-llama3-chat-19b.yaml"
    }

    model_family = list(MODEL_CONFIG.keys())

    model_arch = "cogvlm2"

    def __init__(
        self,
        model_id: str,
        device: str = "cuda:0",
        bf16: bool = True,
        quant: bool = False,
    ):
        super().__init__(model_id)
        TORCH_TYPE = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
        )
        self.device = device
        self.torch_type = TORCH_TYPE

        config = self.MODEL_CONFIG[self.model_id]
        with open(get_abs_path(config)) as f:
            self.model_config = yaml.load(f, Loader=yaml.FullLoader)
        model_path = self.model_config.get("model")["model_path"]

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=self.torch_type,
                trust_remote_code=True,
            )
            .to(self.device)
            .eval()
        )

    @torch.no_grad()
    def chat(self, messages: List, **generation_kwargs):
        assert len(messages) == 1, "Only support one-turn conversation currently"
        text_only_template = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {} ASSISTANT:"
        image = None
        for m in messages:
            if isinstance(m["content"], dict) and "image_path" in m["content"]:
                assert image is None, "not support multi images by now."
                image = Image.open(m["content"]["image_path"]).convert("RGB")

        history = []
        for message in messages:
            if message["role"] in ["system", "user", "assistant"]:
                if isinstance(message["content"], dict):
                    query = message["content"]["text"]
                else:
                    query = message["content"]
            else:
                raise ValueError(
                    "Unsupported role. Only system, user and assistant are supported."
                )

        if image is None:
            query = text_only_template.format(query)
            input_by_model = self.model.build_conversation_input_ids(
                self.tokenizer, query=query, history=history, template_version="chat"
            )
        else:
            input_by_model = self.model.build_conversation_input_ids(
                self.tokenizer,
                query=query,
                history=history,
                images=[image],
                template_version="chat",
            )
        inputs = {
            "input_ids": input_by_model["input_ids"].unsqueeze(0).to(self.device),
            "token_type_ids": input_by_model["token_type_ids"].unsqueeze(0).to(self.device),
            "attention_mask": input_by_model["attention_mask"].unsqueeze(0).to(self.device),
            "images": (
                [[input_by_model["images"][0].to(self.device).to(self.torch_type)]]
                if image is not None
                else None
            ),
        }
        gen_kwargs = {
            "max_new_tokens": 512,
            "pad_token_id": 128002,
            "do_sample": False,
        }
        gen_kwargs.update(generation_kwargs)
        from pprint import pp

        pp(gen_kwargs)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs["input_ids"].shape[1] :]
            response = self.tokenizer.decode(outputs[0])
            response = response.split("<|end_of_text|>")[0]
            print("\nCogVLM2:", response)

        scores = None
        return Response(self.model_id, response, scores, None)

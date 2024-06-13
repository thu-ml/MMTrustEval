
from typing import List
from omegaconf import OmegaConf
import torch

from mmte.utils.registry import registry
from mmte.utils.utils import get_abs_path
from mmte.models.InternVL.internvl_chat_llava.llava.model.builder import load_pretrained_model
from mmte.models.InternVL.internvl_chat_llava.llava.eval.run_llava import chat_model

from mmte.models.base import BaseChat, Response


@registry.register_chatmodel()
class InternVLChat(BaseChat):
    """
    Chat class for InternVL model,
    """

    # TODO: update model config
    MODEL_CONFIG = {
        "InternVL-Chat-ViT-6B-Vicuna-13B": '/data/zhangyichi/Trustworthy-MLLM/MMTrustEval/mmte/configs/models/internvl/internvl_chat_vit_6b_vicuna_13b.yaml',
    }
    model_family = list(MODEL_CONFIG.keys())

    def __init__(self, model_id: str, device: str="cuda:0"):
        super().__init__(model_id)
        self.device = device
        config = self.MODEL_CONFIG[self.model_id]
        self.config = OmegaConf.load(get_abs_path(config))

        self.model_name = self.config.model.model_name
        model_path = self.config.model.model_path

        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=self.model_name,    # get_model_name_from_path(model_path)
            device=self.device
        )
        print(f"Loaded model {self.model_name}")

    @torch.no_grad()
    def chat(self, messages: List, **generation_kwargs):
        # TODO: if system message provided.
        for message in messages:
            if message["role"] in ["system", "user", "assistant"]:
                if message["role"] == "user":
                    if isinstance(message["content"], dict):
                        # multimodal
                        image_path = message["content"]["image_path"]
                        user_message = message["content"]["text"]
                    else:
                        image_path = None
                        user_message = message["content"]
                elif message["role"] == "assistant":
                    # TODO: add assistant answer into the conversation
                    pass
            else:
                raise ValueError("Unsupported role. Only system, user and assistant are supported.")

        if generation_kwargs.get("do_sample") == False:
            temperature = 0.0
            top_p = 1.0
        else:
            # TODO: set the default value
            temperature = generation_kwargs.get("temperature", 0.0)
            top_p = generation_kwargs.get("top_p", 1.0)

        args = type('Args', (), {
            "model_name": self.model_name,
            "query": user_message,
            "conv_mode": None,
            "image_file": image_path,
            "sep": ",",
            "temperature": temperature,
            "top_p": top_p,
            "num_beams": 1,
            "max_new_tokens": generation_kwargs.get("max_new_tokens", 256)
        })()

        output_text = chat_model(self.tokenizer, self.model, self.image_processor, args)
        scores = None
        
        return Response(self.model_id, output_text, scores, None)
    
    

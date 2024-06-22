
from typing import List
from omegaconf import OmegaConf
import cv2
from mmte.utils.registry import registry
from mmte.models.base import BaseChat, Response
from mmte.models.llama_adapter_v2 import llama
from mmte.utils.utils import get_abs_path
import torch
from PIL import Image


@registry.register_chatmodel()
class LlamaAdapterChat(BaseChat):
    """
    Chat class for LlamaAdapterV2 model,
    the origin repo contains: BIAS-7B, LORA-BIAS-7B, CAPTION-7B, we choose BIAS-7B here.
    """

    # TODO: update model config
    MODEL_CONFIG = {
        "llama-adapter-v2": 'configs/models/llama-adapter/llama-adapter-v2-bias-7b.yaml',
    }
    model_family = list(MODEL_CONFIG.keys())

    def __init__(self, model_id: str, device: str="cuda:0"):
        super().__init__(model_id)
        self.device = device
        config = self.MODEL_CONFIG[self.model_id]
        self.config = OmegaConf.load(get_abs_path(config))
        llama_dir = self.config.model.llama_dir
        # if the name is not a path, it will be downloaded from hf
        name = self.config.model.model_path
        self.model, self.preprocess = llama.load(name, llama_dir, device=device)
        if isinstance(self.model, RuntimeError):
            raise RuntimeError(self.model)
        else:
            self.model.eval()


    @torch.no_grad()
    def chat(self, messages: List, **generation_kwargs):
        # TODO: if system message provided.
        for message in messages:
            if message["role"] in ["system", "user", "assistant"]:
                if message["role"] == "user":
                    if isinstance(message["content"], dict):
                        # multimodal
                        image_path = message["content"]["image_path"]
                        user_message = llama.format_prompt(message["content"]["text"])
                        raw_image = Image.fromarray(cv2.imread(image_path))
                        image = self.preprocess(raw_image).unsqueeze(0).to(self.device)
                    else:
                        user_message = message["content"]
                        image = None
                elif message["role"] == "assistant":
                    # TODO: add assistant answer into the conversation
                    pass
            else:
                raise ValueError("Unsupported role. Only system, user and assistant are supported.")
        
        # TODO: list of images/prompts support
        print(user_message)

        # compile the input kwargs and change to max_gen_len, temperature, top_p
        if generation_kwargs.get("do_sample") == False:
            generation_kwargs["temperature"] = 0.0
            # Note: for several models, top_p cannot be 1.0, if any exception raised, use 0.99 instead.
            generation_kwargs["top_p"] = 1.0
        

        # TODO: if we only use do_sample=False, could we set the default value for temperature and top_p to 0.0 and 1.0?        
        generate_kwargs = {
            'max_gen_len': generation_kwargs.get("max_new_tokens", 256),
            'temperature': generation_kwargs.get("temperature", 0.0),
            'top_p': generation_kwargs.get("top_p", 1.0)
        }

        print(generate_kwargs)
        output_text = self.model.generate(imgs=image, prompts=[user_message], **generate_kwargs)
        print(output_text)
        scores = None
        
        return Response(self.model_id, output_text, scores, None)


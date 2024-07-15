
from typing import List
from omegaconf import OmegaConf
import torch

from mmte.utils.registry import registry
from mmte.utils.utils import get_abs_path
from mmte.models.base import BaseChat, Response
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
)
from mmte.models.llava.constants import (
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
)
from mmte.models.llava.model import *
from mmte.models.llava.eval.run_llava import chat_model

from peft import PeftModel
from glob import glob
from huggingface_hub.constants import HF_HUB_CACHE
from huggingface_hub import snapshot_download

@registry.register_chatmodel()
class LLaVARLHFChat(BaseChat):
    """
    Chat class for llava-rlhf model,
    the specific version is LLaVA-RLHF from https://github.com/llava-rlhf/LLaVA-RLHF.
    """

    # TODO: update model config
    MODEL_CONFIG = {
        "llava-rlhf-13b": 'configs/models/llava/llava-rlhf-13b.yaml',
    }
    model_family = list(MODEL_CONFIG.keys())

    def __init__(self, model_id: str, device: str="cuda:0"):
        super().__init__(model_id)
        self.device = device
        config = self.MODEL_CONFIG[self.model_id]
        self.config = OmegaConf.load(get_abs_path(config))
        print(self.config)

        self.model_name = self.config.model.model_name
        download_path = self.config.model.model_path
        snapshot_download(repo_id=download_path, force_download=False)
        model_path = glob("{}/models--zhiqings--LLaVA-RLHF-13b-v1.5-336/snapshots/*/sft_model".format(HF_HUB_CACHE))[0]
        lora_path = glob("{}/models--zhiqings--LLaVA-RLHF-13b-v1.5-336/snapshots/*/rlhf_lora_adapter_model".format(HF_HUB_CACHE))[0]
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        bits = 16
        self.dtype = torch.bfloat16
        self.compute_dtype = torch.bfloat16
        model = LlavaLlamaForCausalLM.from_pretrained(
            model_path,
            device_map={"": self.device},
            torch_dtype=self.dtype,
            load_in_4bit=(bits == 4),
            load_in_8bit=(bits == 8),
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=(bits == 4),
                load_in_8bit=(bits == 8),
                llm_int8_threshold=6.0,
                llm_int8_skip_modules=["mm_projector", "lm_head"],
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=self.compute_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            ),
        )
        
        self.model = PeftModel.from_pretrained(
            model,
            lora_path,
            device_map={"": self.device}
        )

        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            self.tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            self.tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
            )
        model.resize_token_embeddings(len(self.tokenizer))
        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device=self.device, dtype=self.compute_dtype)
        self.image_processor = vision_tower.image_processor

    @torch.no_grad()
    def chat(self, messages: List, **generation_kwargs):
        # TODO: if system message provided.
        assert len(messages) == 1, 'Only support one-turn conversation currently'
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
            "max_new_tokens": generation_kwargs.get("max_new_tokens", 256),
            'dtype': self.dtype
        })()

        output_text = chat_model(self.tokenizer, self.model, self.image_processor, args)
        print(output_text)
        scores = None
        
        return Response(self.model_id, output_text, scores, None)


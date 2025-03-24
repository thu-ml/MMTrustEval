from typing import List

import torch
from transformers import AutoModelForCausalLM

from mmte.models.base import BaseChat, Response
from mmte.models.deepseek_janus.janus.models import (
    MultiModalityCausalLM,
    VLChatProcessor,
)
from mmte.models.deepseek_janus.janus.utils.io import load_pil_images
from mmte.utils.registry import registry


@registry.register_chatmodel()
class DeepSeekJanusChat(BaseChat):
    """
    Chat class for deepseek-janus-7b model,
    """

    # TODO: update model config
    MODEL_CONFIG = {
        "deepseek-janus-7b": 'configs/models/deepseek/deepseek-janus-7b.yaml',
    }
    model_family = list(MODEL_CONFIG.keys())

    def __init__(self, model_id: str, device: str="cuda:0"):
        super().__init__(model_id)
        model_path = "deepseek-ai/Janus-Pro-7B"
        vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
        tokenizer = vl_chat_processor.tokenizer

        vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True
        )
        vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
        
        self.device = device
        self.model = vl_gpt
        self.tokenizer = tokenizer
        self.vl_chat_processor = vl_chat_processor

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
                    else:
                        image_path = None
                        user_message = message["content"]
                elif message["role"] == "assistant":
                    # TODO: add assistant answer into the conversation
                    pass
            else:
                raise ValueError("Unsupported role. Only system, user and assistant are supported.")

        if image_path is not None:
            conversation = [
                {
                    "role": "<|User|>",
                    "content": f"<image_placeholder>\n{user_message}",
                    "images": [image_path],
                },
                {"role": "<|Assistant|>", "content": ""},
            ]
        else:
            conversation = [
                {
                    "role": "<|User|>",
                    "content": f"{user_message}",
                },
                {"role": "<|Assistant|>", "content": ""},
            ]

        pil_images = load_pil_images(conversation)
        prepare_inputs = self.vl_chat_processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True
        ).to(self.model.device)

        # run image encoder to get the image embeddings
        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)

        generation_config = {
            "max_new_tokens": 512,
            "do_sample": False,
            "use_cache": True,
        }
        generation_config.update(generation_kwargs)
        from pprint import pp
        pp(generation_config)
        # run the model to get the response
        outputs = self.model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **generation_config,
        )

        output_text = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)

        scores = None
        return Response(self.model_id, output_text, scores, None)


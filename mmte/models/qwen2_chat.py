from typing import List
import torch
from mmte.models.base import BaseChat, Response
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from mmte.utils.registry import registry
from mmte.models.qwen2_vl.utils.src.qwen_vl_utils import process_vision_info


@registry.register_chatmodel()
class Qwen2Chat(BaseChat):
    """
    Chat class for Qwen2 models
    """

    MODEL_CONFIG = {"qwen2-vl-chat": "Qwen/Qwen2-VL-7B-Instruct"}

    model_family = list(MODEL_CONFIG.keys())

    model_arch = "qwen2"

    def __init__(self, model_id: str, device: str = "cuda:0"):
        super().__init__(model_id)
        config = self.MODEL_CONFIG[self.model_id]
        self.device = device
        self.processor = AutoProcessor.from_pretrained(config)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            config, torch_dtype="auto", device_map="auto"
        )

    @torch.no_grad()
    def chat(self, messages: List, **generation_kwargs):
        for message in messages:
            if message["role"] in ["system", "user", "assistant"]:
                if message["role"] == "user":
                    if isinstance(message["content"], dict):
                        # multimodal
                        image_path = message["content"]["image_path"]
                        text = message["content"]["text"]
                        messages = [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image",
                                        "image": image_path,
                                    },
                                    {"type": "text", "text": text},
                                ],
                            }
                        ]
                        text = self.processor.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                        image_inputs, video_inputs = process_vision_info(messages)
                        inputs = self.processor(
                            text=[text],
                            images=image_inputs,
                            videos=video_inputs,
                            padding=True,
                            return_tensors="pt",
                        )
                        inputs = inputs.to("cuda")
                    else:
                        # text only conversation
                        text = message["content"]
                        messages = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": text},
                                ],
                            }
                        ]
                        text = self.processor.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                        inputs = self.processor(
                            text=[text],
                            padding=True,
                            return_tensors="pt",
                        )
                        inputs = inputs.to("cuda")
                elif message["role"] == "assistant":
                    # TODO: add assistant answer into the conversation
                    pass
            else:
                raise ValueError(
                    "Unsupported role. Only system, user and assistant are supported."
                )
        
        if generation_kwargs.get('max_new_tokens', None) is None:
            generation_kwargs['max_new_tokens'] = 128 
        generated_ids = self.model.generate(**inputs, **generation_kwargs)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return Response(self.model_id, response[0], None, None)

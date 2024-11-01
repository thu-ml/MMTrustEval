from transformers import MllamaForConditionalGeneration, AutoProcessor
from typing import List
from PIL import Image
import torch
from mmte.models.base import BaseChat, Response
from mmte.utils.registry import registry



@registry.register_chatmodel()
class LlamaChat(BaseChat):
    """
    Chat class for llama-3.2 vision model
    """
    
    
    MODEL_CONFIG = {"llama-3-2-chat": "meta-llama/Llama-3.2-11B-Vision-Instruct"}
    
    model_family = list(MODEL_CONFIG.keys())
    
    model_arch = 'llama'
    
    def __init__(self, model_id: str, device: str="cuda:0"):
        super().__init__(model_id)
        config = self.MODEL_CONFIG[self.model_id]
        self.device = device
        self.model = MllamaForConditionalGeneration.from_pretrained(
            config,
            torch_dtype=torch.bfloat16,
            device_map=self.device)
        self.processor = AutoProcessor.from_pretrained(config)
   
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
                            {"role": "user", "content": [
                                {"type": "image"},
                                {"type": "text", "text": text}
                            ]}
                        ]
                        input_text = self.processor.apply_chat_template(
                            messages, add_generation_prompt=True,
                        )
                        inputs = self.processor(
                            Image.open(image_path),
                            input_text,
                            add_special_tokens=False,
                            return_tensors="pt",
                        ).to(self.model.device)
                        

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
                        inputs = inputs.to(self.model.device)
                elif message["role"] == "assistant":
                    # TODO: add assistant answer into the conversation
                    pass
            else:
                raise ValueError("Unsupported role. Only system, user and assistant are supported.")
        
        generation_config = dict(max_new_tokens=512, do_sample=False)
        generation_config.update(generation_kwargs)
        from pprint import pp

        pp(generation_config)
        output = self.model.generate(**inputs, **generation_config)
        prompt_len = inputs.input_ids.shape[-1]
        generated_ids = output[:, prompt_len:]
        response = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return Response(self.model_id, response[0], None, None)
from typing import List
import torch
from mmte.models.base import BaseChat, Response
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoConfig, AutoModel
from mmte.utils.registry import registry
from mmte.models.kosmos2 import Kosmos2ForConditionalGeneration, Kosmos2Config, Kosmos2Model, Kosmos2Processor

AutoConfig.register("kosmos-2", Kosmos2Config)
AutoModel.register(Kosmos2Config, Kosmos2Model)
AutoModelForVision2Seq.register(Kosmos2Config, Kosmos2ForConditionalGeneration)
AutoProcessor.register(Kosmos2Config, Kosmos2Processor)

@registry.register_chatmodel()
class Kosmos2Chat(BaseChat):
    """
    Chat class for Kosmos2 models
    """
    
    MODEL_CONFIG = {"kosmos2-chat": "microsoft/kosmos-2-patch14-224",}
    
    model_family = list(MODEL_CONFIG.keys())
    
    model_arch = 'kosmos2'
    
    def __init__(self, model_id: str, device: str="cuda:0"):
        super().__init__(model_id)
        config = self.MODEL_CONFIG[self.model_id]
        self.device = device
        self.model = AutoModelForVision2Seq.from_pretrained(config)
        self.model.to(self.device)
        self.processor = AutoProcessor.from_pretrained(config)
        
    @torch.no_grad()
    def chat(self, messages: List, **generation_kwargs):
        for message in messages:
            if message["role"] in ["system", "user", "assistant"]:
                if message["role"] == "user":
                    if isinstance(message["content"], dict):
                        # multimodal
                        image_path = message["content"]["image_path"]
                        user_message = message["content"]["text"]
                        image = Image.open(image_path)
                        inputs = self.processor(text=user_message, images=image, return_tensors="pt")
                        inputs = inputs.to(self.device)
                    else:
                        # text only conversation
                        user_message = message["content"]
                        # image = Image.new('RGB', (224, 224))
                        inputs = self.processor(text=user_message, return_tensors="pt")
                        inputs = inputs.to(self.device)
                        inputs["pixel_values"] = None
                        inputs["image_embeds_position_mask"] = None

                elif message["role"] == "assistant":
                    # TODO: add assistant answer into the conversation
                    pass
            else:
                raise ValueError("Unsupported role. Only system, user and assistant are supported.")
            
        generated_ids = self.model.generate(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_embeds=None,
            image_embeds_position_mask=inputs["image_embeds_position_mask"],
            use_cache=True,
            **generation_kwargs
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # By default, the generated  text is cleanup and the entities are extracted.
        processed_text, _ = self.processor.post_process_generation(generated_text)
        response = processed_text.replace(user_message, "").strip()

        return Response(self.model_id, response, None, None)
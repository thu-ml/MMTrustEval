from typing import List
import torch
from mmte.models.base import BaseChat, Response
from mmte.utils.utils import get_abs_path
from omegaconf import OmegaConf
# 假设有BLIP-2模型和处理器的导入路径
from lavis.models import load_model_and_preprocess
from PIL import Image
from mmte.utils.registry import registry

@registry.register_chatmodel()
class InstructlavisBLIPChat(BaseChat):
    """
    Chat class for INSTRUCTBLIP models
    """
    
    MODEL_CONFIG = {
        "instructblip-chat": 'lmsys/vicuna-7b-v1.1',
    }
        
    model_family = list(MODEL_CONFIG.keys())
    
    model_arch = 'instructblip'
    
    def __init__(self, model_id: str, device: str="cuda:0"):
        super().__init__(model_id)
        config = self.MODEL_CONFIG[self.model_id]
        self.device = device
        self.model, self.vis_processors, _ = load_model_and_preprocess(
            name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device=device
        )

    @torch.no_grad()
    def chat(self, messages: List, **generation_kwargs):
        img_list = []
        # TODO: if system message provided.
        for message in messages:
            if message["role"] in ["system", "user", "assistant"]:
                if message["role"] == "user":
                    if isinstance(message["content"], dict):
                        # multimodal
                        image_path = message["content"]["image_path"]
                        user_message = message["content"]["text"]
                        raw_image = Image.open(image_path).convert('RGB')   
                        image = self.vis_processors["eval"](raw_image).unsqueeze(0).to(self.device)
                        output_text = self.model.generate({"image": image, "prompt": user_message})
                    else: 
                        user_message = message["content"]
                        output_text = self.model.generate({"prompt": user_message})
                elif message["role"] == "assistant":
                    # TODO: add assistant answer into the conversation
                    pass
            else:
                raise ValueError("Unsupported role. Only system, user and assistant are supported.")
            
            
        print(output_text)


        scores = None
        
        return Response(self.model_id, output_text, scores, None)
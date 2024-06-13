from typing import List
import torch
from PIL import Image
from omegaconf import OmegaConf
from mmte.models.base import BaseChat, Response
from mmte.utils.utils import get_abs_path
from mmte.utils.registry import registry
from transformers import AutoTokenizer

from mmte.models.mPLUG_Owl.mPLUG_Owl.mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration
from mmte.models.mPLUG_Owl.mPLUG_Owl.mplug_owl.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor

@registry.register_chatmodel()
class mPLUGOwlChat(BaseChat):
    """
    Chat class for mPLUG-Owl models
    """
    
    # TODO: update model config
    MODEL_CONFIG = {
        "mplug-owl-llama-7b": 'configs/models/mplug-owl/mplug-owl-llama-7b.yaml',
    }

    model_family = list(MODEL_CONFIG.keys())
    
    model_arch = 'mplug-owl'
    
    def __init__(self, model_id: str, device: str="cuda:0"):
        super().__init__(model_id)
        config = self.MODEL_CONFIG[self.model_id]
        self.config = OmegaConf.load(get_abs_path(config))
        self.device = device  
        self.model = MplugOwlForConditionalGeneration.from_pretrained(
            self.config.model.model_path,
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        self.image_processor = MplugOwlImageProcessor.from_pretrained(self.config.model.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.model_path)
        self.processor = MplugOwlProcessor(self.image_processor, self.tokenizer)
                
    @torch.no_grad()
    def chat(self, messages: List, **generation_kwargs):
        for message in messages:
            if message["role"] in ["system", "user", "assistant"]:
                if message["role"] == "user":
                    if isinstance(message["content"], dict):
                        # multimodal
                        image_path = message["content"]["image_path"]
                        user_message = message["content"]["text"]
                        raw_image = Image.open(image_path).convert('RGB')  
                        prompts = '''The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
                            Human: <image>
                            Human: {user_message}
                        AI: '''.format(user_message=user_message)
                        inputs = self.processor(text=[prompts], images=[raw_image], return_tensors='pt')
                        inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
                        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                    else: 
                        user_message = message["content"]
                        prompts = '''The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
                            Human: {user_message}
                        AI: '''.format(user_message=user_message)
                        inputs = self.processor(text=[prompts], return_tensors='pt')
                        inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
                        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                elif message["role"] == "assistant":
                    # TODO: add assistant answer into the conversation
                    pass
            else:
                raise ValueError("Unsupported role. Only system, user and assistant are supported.")
            
        generate_kwargs = {
            'do_sample': generation_kwargs.get("do_sample"),
            'top_k': self.config.parameters.top_k,
            'max_length': self.config.parameters.max_length,
            'max_new_tokens': generation_kwargs.get("max_new_tokens"),
        }

        with torch.no_grad():
            res = self.model.generate(**inputs, **generate_kwargs)
        sentence = self.tokenizer.decode(res.tolist()[0], skip_special_tokens=True)


        scores = None
        
        return Response(self.model_id, sentence, scores, None)
























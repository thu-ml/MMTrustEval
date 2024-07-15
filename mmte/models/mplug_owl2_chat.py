from typing import List
import torch
from PIL import Image
from omegaconf import OmegaConf
from mmte.models.base import BaseChat, Response
from mmte.utils.utils import get_abs_path
from mmte.utils.registry import registry
from transformers import TextStreamer
from mmte.models.mPLUG_Owl.mPLUG_Owl2.mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from mmte.models.mPLUG_Owl.mPLUG_Owl2.mplug_owl2.conversation import conv_templates, SeparatorStyle
from mmte.models.mPLUG_Owl.mPLUG_Owl2.mplug_owl2.model.builder import load_pretrained_model
from mmte.models.mPLUG_Owl.mPLUG_Owl2.mplug_owl2.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

@registry.register_chatmodel()
class mPLUGOwl2Chat(BaseChat):
    """
    Chat class for mPLUG-Owl2 models
    """
    
    MODEL_CONFIG = {
        "mplug-owl2-llama2-7b": 'configs/models/mplug-owl2/mplug-owl2-llama2-7b.yaml',
    }

    model_family = list(MODEL_CONFIG.keys())
    
    model_arch = 'mplug-owl2'
    
    def __init__(self, model_id: str, device: str="cuda:0"):
        super().__init__(model_id)
        config = self.MODEL_CONFIG[self.model_id]
        self.config = OmegaConf.load(get_abs_path(config))
        model_path = self.config.model.model_path
        model_name = get_model_name_from_path(model_path)
        self.device = device  
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, None, model_name, load_8bit=False, load_4bit=False, device=self.device
            )
                
    @torch.no_grad()
    def chat(self, messages: List, **generation_kwargs):
        assert len(messages) == 1, 'Only support one-turn conversation currently'
        for message in messages:
            if message["role"] in ["system", "user", "assistant"]:
                if message["role"] == "user":
                    if isinstance(message["content"], dict):
                        # multimodal
                        image_path = message["content"]["image_path"]
                        user_message = message["content"]["text"]
                        raw_image = Image.open(image_path).convert('RGB')  

                        conv = conv_templates["mplug_owl2"].copy()

                        max_edge = max(raw_image.size) # We recommand you to resize to squared image for BEST performance.
                        image = raw_image.resize((max_edge, max_edge))
                        image_tensor = process_images([image], self.image_processor)
                        image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)

                        inp = DEFAULT_IMAGE_TOKEN + user_message
                        conv.append_message(conv.roles[0], inp)
                        conv.append_message(conv.roles[1], None)
                        prompt = conv.get_prompt()

                        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.model.device)
                    else: 
                        user_message = message["content"]
                        image_tensor = None
                        conv = conv_templates["mplug_owl2"].copy()

                        inp = user_message
                        conv.append_message(conv.roles[0], inp)
                        conv.append_message(conv.roles[1], None)
                        prompt = conv.get_prompt()

                        input_ids = tokenizer_image_token(prompt, self.tokenizer, return_tensors='pt').unsqueeze(0).to(self.model.device)
                elif message["role"] == "assistant":
                    # TODO: add assistant answer into the conversation
                    pass
            else:
                raise ValueError("Unsupported role. Only system, user and assistant are supported.")
            
            
        stop_str = conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            if isinstance(message["content"], dict):
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=generation_kwargs.get("do_sample"),
                    temperature=self.config.parameters.temperature,
                    max_new_tokens=generation_kwargs.get("max_new_tokens"),
                    streamer=streamer,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])
            else:
                output_ids = self.model.generate(
                    input_ids,
                    do_sample=generation_kwargs.get("do_sample"),
                    temperature=self.config.parameters.temperature,
                    max_new_tokens=generation_kwargs.get("max_new_tokens"),
                    streamer=streamer,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])

        outputs = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()

        scores = None
        
        return Response(self.model_id, outputs, scores, None)
























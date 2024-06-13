from typing import List
import torch
from PIL import Image
from mmte.models.base import BaseChat, Response
from mmte.utils.utils import get_abs_path
from mmte.utils.registry import registry
from omegaconf import OmegaConf
from mmte.models.ShareGPT4V.share4v.model.builder import load_pretrained_model
from mmte.models.ShareGPT4V.share4v.mm_utils import get_model_name_from_path
from mmte.models.ShareGPT4V.share4v.eval.run_share4v import eval_model
from mmte.models.ShareGPT4V.share4v.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                               DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
from mmte.models.ShareGPT4V.share4v.conversation import SeparatorStyle, conv_templates
from mmte.models.ShareGPT4V.share4v.mm_utils import (KeywordsStoppingCriteria,
                              get_model_name_from_path, tokenizer_image_token)
from mmte.models.ShareGPT4V.share4v.model.builder import load_pretrained_model
from mmte.models.ShareGPT4V.share4v.utils import disable_torch_init


@registry.register_chatmodel()
class ShareGPT4VChat(BaseChat):
    """
    Chat class for ShareGPT4V models
    """
    
    # TODO: update model config
    MODEL_CONFIG = {
        "ShareGPT4V-7B": 'configs/models/sharegpt4v/ShareGPT4V-7B.yaml',
        "ShareGPT4V-13B": 'configs/models/sharegpt4v/ShareGPT4V-13B.yaml',        
    }

    model_family = list(MODEL_CONFIG.keys())
    
    model_arch = 'ShareGPT4V'
    
    def __init__(self, model_id: str, device: str="cuda:0"):
        super().__init__(model_id)
        config = self.MODEL_CONFIG[self.model_id]
        self.config = OmegaConf.load(get_abs_path(config))
        self.device = device  
        # print(self.config.model.model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path=self.config.model.model_path,
            model_base=None,
            model_name=get_model_name_from_path(self.config.model.model_path),
            device=self.device
        )
                
    @torch.no_grad()
    def chat(self, messages: List, **generation_kwargs):
        for message in messages:
            if message["role"] in ["system", "user", "assistant"]:
                if message["role"] == "user":
                    if isinstance(message["content"], dict):
                        # multimodal
                        image_path = message["content"]["image_path"]
                        user_message = message["content"]["text"]

                        qs = user_message
                        if self.model.config.mm_use_im_start_end:
                            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + \
                                DEFAULT_IM_END_TOKEN + '\n' + qs
                        else:
                            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

                        conv_mode = "share4v_v0"
                        conv = conv_templates[conv_mode].copy()
                        conv.append_message(conv.roles[0], qs)
                        conv.append_message(conv.roles[1], None)
                        prompt = conv.get_prompt()

                        raw_image = Image.open(image_path).convert('RGB')
                        image_tensor = self.image_processor.preprocess(raw_image, return_tensors='pt')['pixel_values'].half().to(self.device)
                        input_ids = tokenizer_image_token(
                            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
                    else: 
                        image_tensor = None
                        user_message = message["content"]
                        qs = user_message
                        if self.model.config.mm_use_im_start_end:
                            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + \
                                DEFAULT_IM_END_TOKEN + '\n' + qs
                        else:
                            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

                        conv_mode = "share4v_v0"
                        conv = conv_templates[conv_mode].copy()
                        conv.append_message(conv.roles[0], qs)
                        conv.append_message(conv.roles[1], None)
                        prompt = conv.get_prompt()
                        inputs = self.tokenizer([prompt])
                        input_ids = torch.as_tensor(inputs.input_ids).to(self.device)
                elif message["role"] == "assistant":
                    # TODO: add assistant answer into the conversation
                    pass
            else:
                raise ValueError("Unsupported role. Only system, user and assistant are supported.")
            
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(
            keywords, self.tokenizer, input_ids)
        
        with torch.inference_mode():
            if isinstance(message["content"], dict):
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=generation_kwargs.get("do_sample"),
                    temperature=0.2,
                    max_new_tokens=generation_kwargs.get("max_new_tokens"),
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])
            else:
                output_ids = self.model.generate(
                    input_ids,
                    do_sample=generation_kwargs.get("do_sample"),
                    temperature=0.2,
                    max_new_tokens=generation_kwargs.get("max_new_tokens"),
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (
            input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(
                f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = self.tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        scores = None
        
        return Response(self.model_id, outputs, scores, None)
























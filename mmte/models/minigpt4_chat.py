from typing import List
import torch
from mmte.utils.registry import registry
from mmte.models.base import BaseChat, Response
from mmte.utils.utils import get_abs_path
from omegaconf import OmegaConf
from mmte.models.minigpt4.models import MiniGPT4
from mmte.models.minigpt4.processors import Blip2ImageEvalProcessor
from mmte.models.minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2, StoppingCriteriaSub
from transformers import StoppingCriteriaList
from copy import deepcopy


@registry.register_chatmodel()
class MiniGPT4Chat(BaseChat):
    """
    Chat class for MiniGPT4 models, e.g., minigpt-4-vicuna-7b-v0
    """
    
    MODEL_CONFIG = {"minigpt-4-vicuna-13b-v0": 'configs/models/minigpt-4/minigpt-4-vicuna-13b-v0.yaml',
                    "minigpt-4-llama2-7b": 'configs/models/minigpt-4/minigpt-4-llama2-7b.yaml'}
    
    CONV_DICT = {"minigpt-4-vicuna-13b-v0": CONV_VISION_Vicuna0,
                 "minigpt-4-llama2-7b": CONV_VISION_LLama2}
    
    model_family = list(MODEL_CONFIG.keys())
    
    model_arch = 'minigpt-4'
    
    def __init__(self, model_id: str, device: str="cuda:0"):
        super().__init__(model_id)
        config = self.MODEL_CONFIG[self.model_id]
        self.config = OmegaConf.load(get_abs_path(config))
        self.device = device
        self.model = MiniGPT4.from_config(self.config.model).to(self.device).eval()
        self.vis_processor = Blip2ImageEvalProcessor.from_config(self.config.preprocess.vis_processor.eval)
        
        stop_words_ids = [[835], [2277, 29937]]
        stop_words_ids = [torch.tensor(ids).to(device=self.device) for ids in stop_words_ids]
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
        
        self.client = Chat(self.model, self.vis_processor, device=self.device, stopping_criteria=self.stopping_criteria)
        
    @torch.no_grad()
    def chat(self, messages: List, **generation_kwargs):
        conversation = self.CONV_DICT[self.model_id].copy()
        img_list = []
        # TODO: if system message provided.
        for message in messages:
            if message["role"] in ["system", "user", "assistant"]:
                if message["role"] == "user":
                    if isinstance(message["content"], dict):
                        # multimodal
                        image_path = message["content"]["image_path"]
                        user_message = message["content"]["text"]
                        self.client.upload_img(image_path, conversation, img_list)
                        # TODO: encode img support multiple images, change index
                        self.client.encode_img(img_list)
                        self.client.ask(user_message, conversation)
                    else:
                        user_message = message["content"]
                        self.client.ask(user_message, conversation)
                elif message["role"] == "assistant":
                    # TODO: add assistant answer into the conversation
                    self.client.assistant_history(message["content"], conversation)
            else:
                raise ValueError("Unsupported role. Only system, user and assistant are supported.")
            
            
        # Generation Configuration
        if "stop_sequences" in generation_kwargs:
            if isinstance(generation_kwargs.get("stop_sequences"), str):
                generation_kwargs["stop_sequences"] = [generation_kwargs["stop_sequences"]]
            
            customized_stop_ids = []
            for stop in generation_kwargs.get("stop_sequences"):
                customized_stop_ids.append(self.model.llama_tokenizer(stop, return_tensors='pt', add_special_tokens=False).input_ids.to(self.device))
            
            generation_kwargs["stopping_criteria"] = deepcopy(self.stopping_criteria).append(StoppingCriteriaSub(customized_stop_ids))
            generation_kwargs.pop("stop_sequences")
            
            
        output = self.client.answer(conv=conversation, img_list=img_list, **generation_kwargs)
        
        output_text = self.model.llama_tokenizer.decode(output.sequences[0], skip_special_tokens=True)
        output_text = output_text.split('###')[0]  # remove the stop sign '###'
        output_text = output_text.split('Assistant:')[-1].strip()

        scores = None
        if "scores" in output.keys() and output.scores[0].shape[0]==1:
            scores = torch.cat(output.scores).cpu().numpy()
        
        return Response(self.model_id, output_text, scores, None)
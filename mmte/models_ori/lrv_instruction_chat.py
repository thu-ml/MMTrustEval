from typing import List
import torch
from mmte.models.base import BaseChat, Response
from mmte.utils.utils import get_abs_path
from mmte.utils.registry import registry
from omegaconf import OmegaConf
from transformers import StoppingCriteriaList
from copy import deepcopy
from mmte.models.LRV_Instruction.MiniGPT_4.minigpt4.common.config import Config
from mmte.models.LRV_Instruction.MiniGPT_4.minigpt4.common.registry import registry as registry_minigpt4
from mmte.models.LRV_Instruction.MiniGPT_4.minigpt4.conversation.conversation import Chat, CONV_VISION, StoppingCriteriaSub

# imports modules for registration
from mmte.models.LRV_Instruction.MiniGPT_4.minigpt4.datasets.builders import *
from mmte.models.LRV_Instruction.MiniGPT_4.minigpt4.models import *
from mmte.models.LRV_Instruction.MiniGPT_4.minigpt4.processors import *
from mmte.models.LRV_Instruction.MiniGPT_4.minigpt4.runners import *
from mmte.models.LRV_Instruction.MiniGPT_4.minigpt4.tasks import *

@registry.register_chatmodel()
class LRVInstructionChat(BaseChat):
    """
    Chat class for MiniGPT4 models, e.g., minigpt-4-vicuna-7b-v0
    """
    
    MODEL_CONFIG = {"lrv-instruction": 'configs/models/lrv-instruction/minigpt4_eval.yaml'}
    
    CONV_DICT = {"lrv-instruction": CONV_VISION}
    
    model_family = list(MODEL_CONFIG.keys())
    
    def __init__(self, model_id: str, device: str="cuda:0"):
        super().__init__(model_id)
        config = self.MODEL_CONFIG[self.model_id]
        self.config = OmegaConf.load(get_abs_path(config))
        self.device = device
        self.cfg_path = "configs/models/lrv-instruction/minigpt4_eval.yaml"
        self.options = []
        cfg = Config(get_abs_path(self.cfg_path))
        model_config = cfg.model_cfg
        model_config.device_8bit = 0
        model_cls = registry_minigpt4.get_model_class(model_config.arch)
        self.model = model_cls.from_config(model_config).to(self.device)

        vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        self.vis_processor = registry_minigpt4.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
        
        stop_words_ids = [[835], [2277, 29937]]
        stop_words_ids = [torch.tensor(ids).to(device=self.device) for ids in stop_words_ids]
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
        self.client = Chat(self.model, self.vis_processor, device=self.device)

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
            
            
        output_text, _ = self.client.answer(conv=conversation, img_list=img_list, **generation_kwargs)
        scores = None

        return Response(self.model_id, output_text, scores, None)
from mmte.utils.registry import registry
from mmte.models.base import BaseChat, Response
from mmte.models.llava_chat import LLaVAChat
from mmte.models.llava_rlhf_chat import LLaVARLHFChat
from mmte.models.llava_hf_chat import LLaVAHFChat
from mmte.models.llava_hf_next_chat import LLaVANEXTChat
from mmte.models.mplug_owl_chat import mPLUGOwlChat
from mmte.models.mplug_owl2_chat import mPLUGOwl2Chat
from mmte.models.mplug_owl3_chat import mPLUGOwl3Chat
from mmte.models.internvl2_chat import InternVL2Chat
from mmte.models.openai_chat import OpenAIChat
from mmte.models.google_chat import GoogleChat
from mmte.models.claude3_chat import ClaudeChat
from mmte.models.qwen_plus_chat import QwenPlusChat
from mmte.models.minigpt4_chat import MiniGPT4Chat
from mmte.models.instructblip_chat import InstructBLIPChat
from mmte.models.qwen_chat import QwenChat
from mmte.models.qwen2_chat import Qwen2Chat
from mmte.models.otter_chat import OtterChat
from mmte.models.internlm_xcomposer_chat import InternLMXComposerChat
from mmte.models.cogvlm_chat import CogVLMChat
from mmte.models.phi3_chat import Phi3Chat
from mmte.models.deepseek_chat import DeepSeekVL
from mmte.models.hunyuan_chat import HunyuanChat
from mmte.models.llama3_2 import LlamaChat
from typing import List


def load_chatmodel(model_id: str, device: str = "cuda:0") -> "BaseChat":
    return registry.get_chatmodel_class(model_id)(model_id=model_id, device=device)


def model_zoo() -> List["BaseChat"]:
    return registry.list_chatmodels()

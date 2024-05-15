from typing import List, Dict, Any, Literal
import google.generativeai as genai
from google.generativeai.types import HarmBlockThreshold, HarmCategory, GenerationConfig
from google.generativeai.types.answer_types import to_finish_reason
import yaml
from mmte.utils.registry import registry
from mmte.models.base import BaseChat, Response
from mmte.utils.utils import get_abs_path
import os
from PIL import Image
import time

@registry.register_chatmodel()
class GoogleChat(BaseChat):
    """
    Chat class for Google models, e.g., gemini-pro-vision
    """
    
    MODEL_CONFIG = {"gemini-pro": 'configs/models/google/google.yaml',
                    "gemini-pro-vision": 'configs/models/google/google.yaml'}
    
    model_family = list(MODEL_CONFIG.keys())
    
    model_arch = 'gemini'
    
    def __init__(self, model_id: str="gemini-pro-vision", **kargs):
        super().__init__(model_id=model_id)
        config = self.MODEL_CONFIG[self.model_id]
        with open(get_abs_path(config)) as f:
            self.model_config = yaml.load(f, Loader=yaml.FullLoader)
        # use_proxy = self.model_config.get('proxy')
        # if use_proxy is not None:
        #     os.environ['http_proxy'] = self.model_config.get('proxy')
        #     os.environ['https_proxy'] = self.model_config.get('proxy')
        self.api_key = self.model_config.get('api_key')
        genai.configure(api_key=self.api_key)
        self.max_retries = self.model_config.get('max_retries', 10)
        self.timeout = self.model_config.get('timeout', 1)
        self.client = genai.GenerativeModel(self.model_id)
        
        safety_config = self.model_config.get("safety_settings")
        self.safety_settings = None
        if safety_config is not None:
            self.safety_settings = {}
            for harm_category in safety_config:
                self.safety_settings[HarmCategory[harm_category]] = HarmBlockThreshold[safety_config[harm_category]]
        print("Safety Setting for Gemini: ", self.safety_settings)
        
        
    def generate_content(self, conversation, generation_config):
        response = None
        for i in range(self.max_retries):
            try:
                response = self.client.generate_content(conversation, generation_config=generation_config, safety_settings=self.safety_settings if self.safety_settings is not None else None)
                break
            except Exception as e:
                print(f"Error in generation: {e}")
                response = f"Error in generation: {e}"
                time.sleep(self.timeout)
        return response
        
        
    def chat(self, messages: List, **generation_kwargs):
        conversation = []
        for message in messages:
            if message["role"] in ["system", "user", "assistant"]:
                if isinstance(message['content'], dict):
                    
                    # multimodal content
                    text = message['content']['text']
                    image_path = message['content']['image_path']
                    if os.path.exists(image_path):
                        local_image = image_path
                    image = Image.open(local_image)
                    
                    content = [text, image]
                else:
                    content = [message['content']]
                if message["role"] == "assistant":
                    message["role"] = "model"
                conversation.append({"role": message["role"], "parts": content})
            else:
                raise ValueError("Unsupported role. Only system, user and assistant are supported.")
        
        generation_config = {}
        if "stop_sequences" in generation_kwargs:
            generation_config["stop_sequences"] = generation_kwargs.get("stop_sequences")
        generation_config["max_output_tokens"] = generation_kwargs.get("max_new_tokens", 100)
        # generation_config["max_output_tokens"] = None
        if "do_sample" in generation_kwargs and not generation_kwargs.get("do_sample"):
            generation_config["temperature"] = 0.0
        generation_config = GenerationConfig(**generation_config)
        response = self.generate_content(conversation, generation_config)
        if isinstance(response, str):
            return Response(self.model_id, response, None, None)

        # response_message = response.text
        response_message = ""
        finish_reason = "unknown"
        try:
            response_message = response.candidates[0].content.parts[0].text
            finish_reason = to_finish_reason(response.candidates[0].finish_reason)
        except Exception as e:
            print(f"Error in response extraction: {e}")
            generation_config.max_output_tokens = None
            response = self.generate_content(conversation, generation_config)
            if isinstance(response, str):
                return Response(self.model_id, response, None, None)

            try:
                response_message = response.candidates[0].content.parts[0].text
                finish_reason = to_finish_reason(response.candidates[0].finish_reason)
            except Exception as e:
                print(f"Error in response extraction: {e}")
                pass
            
        
        
        return Response(self.model_id, response_message, None, finish_reason)

    
    # Function to encode the image
    @classmethod
    def encode_image(cls, image_path: str):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    
    

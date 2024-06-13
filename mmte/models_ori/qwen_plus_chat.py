from http import HTTPStatus
from typing import List, Dict, Any
import dashscope
import yaml
import os
from PIL import Image
from mmte.utils.registry import registry
from mmte.models.base import BaseChat, Response
from mmte.utils.utils import get_abs_path

def transform_messages(messages):
    transformed_messages = []
    for message in messages:
        # 提取当前消息的角色和内容
        role = message.get('role')
        if isinstance(message.get('content'), dict):
            content_dict = message.get('content')
            # 初始化一个空列表来存放转换后的内容项
            transformed_content = []
            # 检查是否有文本内容并添加到列表
            if 'text' in content_dict:
                transformed_content.append({"text": content_dict['text']})
            # 检查是否有图像路径并添加到列表
            if 'image_path' in content_dict:
                transformed_content.append({"image": content_dict['image_path']})
        else:
            transformed_content = []
            transformed_content.append({"text": message.get('content')})
            # 将转换后的内容列表与角色组合成新的消息格式
        transformed_messages.append({
            "role": role,
            "content": transformed_content
        })

    return transformed_messages

@registry.register_chatmodel()
class QwenPlusChat(BaseChat):
    """
    Chat class for Qwen-Plus API, handling multimodal generation.
    """
    
    MODEL_CONFIG = {"qwen-vl-plus": 'configs/models/qwen/qwen.yaml'}
    
    model_family = list(MODEL_CONFIG.keys())
    
    model_arch = 'qwen-plus'
    
    def __init__(self, model_id: str="qwen-vl-plus", **kargs):
        super().__init__(model_id=model_id)
        config = self.MODEL_CONFIG[self.model_id]
        with open(get_abs_path(config)) as f:
            self.model_config = yaml.load(f, Loader=yaml.FullLoader)
        self.api_key = self.model_config.get('api_key')


    def chat(self, messages: List[Dict[str, Any]], **generation_kwargs):
        """Simple single round multimodal conversation call.
        """

        messages = transform_messages(messages)
        response = dashscope.MultiModalConversation.call(model='qwen-vl-plus',
                                                        messages=messages)
        # The response status_code is HTTPStatus.OK indicate success,
        # otherwise indicate request is failed, you can get error code
        # and message from code and message.
        if response.status_code == HTTPStatus.OK:                    
            response_message = response.output.choices[0].message.content[0]['text']
            finish_reason = response.output.choices[0].finish_reason
            logprobs = None
            return Response(self.model_id, response_message, logprobs, finish_reason)
        else:
            print(response.code)  # The error code.
            print(response.message)  # The error message.
            return Response(self.model_id, "API FAILED", None, None)

    

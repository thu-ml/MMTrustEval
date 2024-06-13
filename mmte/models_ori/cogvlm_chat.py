from typing import List, Optional, Tuple, Literal
import torch
from mmte.utils.registry import registry
from mmte.models.base import BaseChat, Response
from mmte.utils.utils import get_abs_path
from PIL import Image
import yaml
from transformers import AutoModelForCausalLM, LlamaTokenizer
from torchvision import transforms

LANGUAGE_TOKEN_TYPE = 0
VISION_TOKEN_TYPE = 1


@registry.register_chatmodel()
class CogVLMChat(BaseChat):
    """
    Chat class for CogVLM models
    """
    
    MODEL_CONFIG = {"cogvlm-chat-hf": "configs/models/cogvlm/cogvlm-chat-hf.yaml"}
    
    model_family = list(MODEL_CONFIG.keys())
    
    model_arch = 'cogvlm'
    
    def __init__(self, model_id: str, device: str="cuda:0", bf16: bool=True, quant: bool=False):
        super().__init__(model_id)
        config = self.MODEL_CONFIG[self.model_id]
        with open(get_abs_path(config)) as f:
            self.model_config = yaml.load(f, Loader=yaml.FullLoader)
        if bf16:
            self.torch_type = torch.bfloat16
        else:
            self.torch_type = torch.float16
        model_path = self.model_config.get('from_pretrained')
        tokenizer_path = self.model_config.get('local_tokenizer')
        
        self.device = device
        
        self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
        if quant:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=self.torch_type,
                low_cpu_mem_usage=True,
                load_in_4bit=True,
                trust_remote_code=True
            ).eval()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=self.torch_type,
                low_cpu_mem_usage=True,
                load_in_4bit=False,
                trust_remote_code=True
            ).to(self.device).eval()
            
            
    def _history_to_prompt(self, signal_type, history):
        if signal_type == 'base':
            prompt = ''
            for i, (role, message) in enumerate(history):
                if i==0 and role=='system':
                    prompt += message + " "
                if role=='user':
                    prompt += "USER: {} ".format(message)
                elif role=='assistant':
                    prompt += "ASSISTANT: {}\n".format(message)
            prompt += "ASSISTANT:"
            return prompt
        elif signal_type == 'vqa':
            answer_format = 'Short answer:'
        elif signal_type == 'chat':
            answer_format = 'Answer:'
        else:
            assert False, f"Unknown signal type {signal_type}"
            
        prompt = ''
        for i, (role, message) in enumerate(history):
            if i==0 and role=='system':
                prompt += message + " "
            if role=='user':
                prompt += "Question: {} ".format(message)
            elif role=='assistant':
                prompt += "{} {}\n".format(answer_format, message)
        prompt += answer_format
        return prompt
            
            
    def build_conversation_input_ids(
            self,
            *,
            history: Optional[List[Tuple[str, str]]] = None,
            images: Optional[List["PIL.Image"]] = None,
            template_version: Optional[Literal["base", "chat", "vqa"]] = None,
    ):
        image_size: int = self.model.config.vision_config['image_size']
        patch_size: int = self.model.config.vision_config['patch_size']
        template_version = template_version or self.model.config.template_version
        assert images is None or len(images) <= 1, f"not support multi images by now."
        history = history or []
        text = self._history_to_prompt(template_version, history)

        input_ids = [self.tokenizer.bos_token_id]
        token_type_ids = [LANGUAGE_TOKEN_TYPE]
        if images is not None and len(images) == 1:
            # vision
            transform = transforms.Compose(
                [
                    transforms.Resize(
                        (image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                ]
            )
            images = [transform(images[0])]
            # language
            vision_token_num = (image_size // patch_size) * (image_size // patch_size) + 2
            input_ids += [self.tokenizer.pad_token_id] * vision_token_num
            token_type_ids += [VISION_TOKEN_TYPE] * vision_token_num
        text_ids = self.tokenizer.encode(text, add_special_tokens=False)

        input_ids += text_ids
        token_type_ids += [LANGUAGE_TOKEN_TYPE] * len(text_ids)
        attention_mask = [1] * len(input_ids)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'images': images,
        }
            
            
    
    @torch.no_grad()
    def chat(self, messages: List, **generation_kwargs):
        text_only_template = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
        image = None
        for m in messages:
            if isinstance(m['content'], dict) and 'image_path' in m['content']:
                assert image is None, "not support multi images by now."
                image = Image.open(m['content']['image_path']).convert('RGB')
        
        history = []
        
        if image is None and messages[0]['role']!='system':
            messages = [{"role": "system", "content": text_only_template}]+messages
        
        for message in messages:
            if message["role"] in ["system", "user", "assistant"]:
                if isinstance(message["content"], dict):
                    history.append((message['role'], message['content']['text']))
                else:
                    history.append((message['role'], message['content']))
            else:
                raise ValueError("Unsupported role. Only system, user and assistant are supported.")
        
        if image is None:
            input_by_model = self.build_conversation_input_ids(history=history, template_version='base')
        else:
            input_by_model = self.build_conversation_input_ids(history=history, images=[image])

        inputs = {
            'input_ids': input_by_model['input_ids'].unsqueeze(0).to(self.device),
            'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(self.device),
            'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(self.device),
            'images': [[input_by_model['images'][0].to(self.device).to(self.torch_type)]] if image is not None else None,
        }
        
        # add any transformers params here.
        gen_kwargs = {"max_length": 2048,
                      "do_sample": False,
                      "return_dict_in_generate": True} # "temperature": 0.9
        gen_kwargs.update(generation_kwargs)
        outputs = self.model.generate(**inputs, **gen_kwargs)
        output_text = outputs.sequences
        output_text = output_text[:, inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(output_text[0])
        response = response.split("</s>")[0]
        
        scores = None
        if "scores" in outputs.keys() and outputs.scores[0].shape[0]==1:
            scores = torch.cat(outputs.scores).cpu().numpy()
            
        return Response(self.model_id, response, scores, None)
                    
        
        
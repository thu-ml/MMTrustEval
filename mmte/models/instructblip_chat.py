from typing import List
from mmte.models.base import BaseChat, Response
from mmte.utils.registry import registry
from PIL import Image
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch

@registry.register_chatmodel()
class InstructBLIPChat(BaseChat):
    """
    Chat class for INSTRUCTBLIP models
    """
    
    MODEL_CONFIG = {
        "instructblip-vicuna-7b": 'Salesforce/instructblip-vicuna-7b',
        "instructblip-vicuna-13b": 'Salesforce/instructblip-vicuna-13b',
        "instructblip-flan-t5-xxl": 'Salesforce/instructblip-flan-t5-xxl'
    }
        
    model_family = list(MODEL_CONFIG.keys())
    
    model_arch = 'instructblip'
    
    def __init__(self, model_id: str, device: str="cuda:0"):
        super().__init__(model_id)
        config = self.MODEL_CONFIG[self.model_id]
        self.device = device
        self.model = InstructBlipForConditionalGeneration.from_pretrained(self.MODEL_CONFIG[model_id]).to(self.device)
        self.processor = InstructBlipProcessor.from_pretrained(self.MODEL_CONFIG[model_id])

    @torch.no_grad()
    def chat(self, messages: List, **generation_kwargs):
        img_list = []
        multimodal = False
        # TODO: if system message provided.
        for message in messages:
            if message["role"] in ["system", "user", "assistant"]:
                if message["role"] == "user":
                    if isinstance(message["content"], dict):
                        # multimodal
                        multimodal = True
                        image_path = message["content"]["image_path"]
                        user_message = message["content"]["text"]
                        raw_image = Image.open(image_path).convert('RGB')  
                        inputs = self.processor(images=raw_image, text=user_message, return_tensors="pt").to(self.device)
                    else: 
                        user_message = message["content"]
                        inputs = self.processor(text=user_message, return_tensors="pt").to(self.device)
                        print(inputs)
                elif message["role"] == "assistant":
                    # TODO: add assistant answer into the conversation
                    pass
            else:
                raise ValueError("Unsupported role. Only system, user and assistant are supported.")
            
            
            if multimodal:
                outputs = self.model.generate(
                    **inputs,
                    do_sample=generation_kwargs.get("do_sample"),
                    num_beams=5,
                    max_new_tokens=generation_kwargs.get("max_new_tokens"),
                    min_length=1,
                    repetition_penalty=1.5,
                    length_penalty=1.0,
                    temperature=1,
                )
            
            else:
                # concatenate query embeddings with prompt embeddings
                inputs_embeds = self.model.get_input_embeddings()(inputs.input_ids)
                attention_mask = inputs.attention_mask
                
                outputs = self.model.language_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    do_sample=generation_kwargs.get("do_sample"),
                    num_beams=5,
                    max_new_tokens=generation_kwargs.get("max_new_tokens"),
                    min_length=1,
                    repetition_penalty=1.5,
                    length_penalty=1.0,
                    temperature=1,
                )
                
                if not self.model.language_model.config.is_encoder_decoder:
                    # the InstructBLIP authors used inconsistent tokenizer/model files during training,
                    # with the tokenizer's bos token being set to </s> which has ID=2,
                    # whereas the model's text config has bos token id = 0
                    bos_token_id = (
                        2
                        if self.model.config.text_config.architectures[0] == "LLaMAForCausalLM"
                        else self.model.config.text_config.bos_token_id
                    )
                    bos_tokens = torch.LongTensor([[bos_token_id]]).repeat(1, 1).to(self.device)
                    if not isinstance(outputs, torch.Tensor):
                        outputs.sequences = torch.cat([bos_tokens, outputs.sequences], dim=-1)
                    else:
                        outputs = torch.cat([bos_tokens, outputs], dim=-1)
                
            generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        scores = None
        
        return Response(self.model_id, generated_text, scores, None)
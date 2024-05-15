from typing import List
import torch
from mmte.models.base import BaseChat, Response
from mmte.models.mic.instructblip import InstructBlipConfig, InstructBlipForConditionalGeneration, InstructBlipProcessor
from PIL import Image
from mmte.utils.registry import registry


@registry.register_chatmodel()
class MMICLChat(BaseChat):
    """
    Chat class for MMICL models
    """
    
    MODEL_CONFIG = {"mmicl-instructblip-t5-xxl-chat": {"processor": "Salesforce/instructblip-flan-t5-xxl", "model": "BleachNick/MMICL-Instructblip-T5-xxl"},}
    
    model_family = list(MODEL_CONFIG.keys())
    
    model_arch = 'mmicl'
    
    def __init__(self, model_id: str, device: str="cuda:0"):
        super().__init__(model_id)
        self.device = device
        config = self.MODEL_CONFIG[self.model_id]
        model_config = config['model']
        processor_config = config['processor']
        self.config = InstructBlipConfig.from_pretrained(model_config)
        self.model = InstructBlipForConditionalGeneration.from_pretrained(
            model_config,
            config=self.config).to(self.device, dtype=torch.bfloat16)

        self.processor = InstructBlipProcessor.from_pretrained(processor_config)
        image_palceholder="图"
        sp = [image_palceholder] + [f"<image{i}>" for i in range(20)]
        sp = sp + self.processor.tokenizer.additional_special_tokens[len(sp):]
        self.processor.tokenizer.add_special_tokens({'additional_special_tokens': sp})
        if self.model.qformer.embeddings.word_embeddings.weight.shape[0] != len(self.processor.qformer_tokenizer):
            self.model.qformer.resize_token_embeddings(len(self.processor.qformer_tokenizer))
        self.replace_token="".join(32*[image_palceholder])

        
    @torch.no_grad()
    def chat(self, messages: List, **generation_kwargs):
        for message in messages:
            if message["role"] in ["system", "user", "assistant"]:
                if message["role"] == "user":
                    if isinstance(message["content"], dict):
                        # multimodal
                        image_path = message["content"]["image_path"]
                        user_message = message["content"]["text"]
                        image = Image.open(image_path)
                        images = [image]
                        prompt = [f'Use the image 0: <image0>{self.replace_token} as a visual aid to help you answer the question accurately. {user_message}']
                        inputs = self.processor(images=images, text=prompt, return_tensors="pt")
                        inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
                        inputs['img_mask'] = torch.tensor([[1 for i in range(len(images))]])
                        inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)
                        inputs = inputs.to(self.device)
                    else:
                        # text only conversation
                        user_message = message["content"]
                        prompt = [user_message]
                        inputs = self.processor(images=None, text=prompt, return_tensors="pt")
                        inputs = inputs.to(self.device)
                        inputs['pixel_values'] = None
                        inputs['img_mask'] = None


                elif message["role"] == "assistant":
                    # TODO: add assistant answer into the conversation
                    pass
            else:
                raise ValueError("Unsupported role. Only system, user and assistant are supported.")
        
        if 'max_new_tokens' in generation_kwargs:
            max_new_tokens = generation_kwargs.pop('max_new_tokens')
            generation_kwargs['max_length'] = max_new_tokens
        if 'min_new_tokens' in generation_kwargs:
            min_new_tokens = generation_kwargs.pop('min_new_tokens')
            generation_kwargs['min_length'] = min_new_tokens

        outputs = self.model.generate(
            pixel_values = inputs['pixel_values'],
            input_ids = inputs['input_ids'],
            attention_mask = inputs['attention_mask'],
            img_mask = inputs['img_mask'],
            **generation_kwargs
        )
        response = self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()

        return Response(self.model_id, response, None, None)
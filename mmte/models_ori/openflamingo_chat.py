from typing import List
from open_flamingo import create_model_and_transforms
from PIL import Image
import requests
import torch
from omegaconf import OmegaConf
from mmte.models.base import BaseChat, Response
from mmte.utils.utils import get_abs_path
from mmte.utils.registry import registry
from huggingface_hub import hf_hub_download

@registry.register_chatmodel()
class OpenFlamingoChat(BaseChat):
    """
    Chat class for  OpenFlamingo models
    """
    
    # TODO: update model config
    MODEL_CONFIG = {
        "OpenFlamingo-3B-vitl-mpt1b": 'configs/models/openflamingo/OpenFlamingo-3B-vitl-mpt1b.yaml',
    }

    model_family = list(MODEL_CONFIG.keys())
    
    model_arch = 'openflamingo'
    
    def __init__(self, model_id: str, device: str="cuda:0"):
        super().__init__(model_id)
        config = self.MODEL_CONFIG[self.model_id]
        self.config = OmegaConf.load(get_abs_path(config))
        self.device = device  
        self.model, self.image_processor, self.tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
            tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
            cross_attn_every_n_layers=1,
        )
        checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-3B-vitl-mpt1b", "checkpoint.pt")
        self.model.load_state_dict(torch.load(checkpoint_path), strict=False)
        self.model.to(self.device)

    @torch.no_grad()
    def chat(self, messages: List, **generation_kwargs):
        for message in messages:
            if message["role"] in ["system", "user", "assistant"]:
                if message["role"] == "user":
                    if isinstance(message["content"], dict):
                        # multimodal
                        image_path = message["content"]["image_path"]
                        user_message = message["content"]["text"]
                        raw_image = Image.open(image_path).convert('RGB')  

                        # demo_image_one = Image.open(
                        #     requests.get(
                        #         "http://images.cocodataset.org/val2017/000000039769.jpg", stream=True
                        #     ).raw
                        # )

                        # demo_image_two = Image.open(
                        #     requests.get(
                        #         "http://images.cocodataset.org/test-stuff2017/000000028137.jpg",
                        #         stream=True
                        #     ).raw
                        # )

                        # query_image = Image.open(
                        #     requests.get(
                        #         "http://images.cocodataset.org/test-stuff2017/000000028352.jpg", 
                        #         stream=True
                        #     ).raw
                        # )

                        vision_x = [self.image_processor(raw_image).unsqueeze(0)]
                        vision_x = torch.cat(vision_x, dim=0)
                        vision_x = vision_x.unsqueeze(1).unsqueeze(0)

                        self.tokenizer.padding_side = "left" # For generation padding tokens should be on the left
                        lang_x = self.tokenizer(
                            ["<image>{}".format(user_message)],
                            return_tensors="pt",
                        )
                    else: 
                        pass
                elif message["role"] == "assistant":
                    # TODO: add assistant answer into the conversation
                    pass
            else:
                raise ValueError("Unsupported role. Only system, user and assistant are supported.")
        
        generated_text = self.model.generate(
            vision_x=vision_x.to(self.device),
            lang_x=lang_x["input_ids"].to(self.device),
            attention_mask=lang_x["attention_mask"].to(self.device),
            max_new_tokens=generation_kwargs.get("max_new_tokens"),
            num_beams=3,
            do_sample=generation_kwargs.get("do_sample"),
        )

        print("Generated text: ", self.tokenizer.decode(generated_text[0]).split("{}".format(user_message))[1].split("<")[0])

        scores = None
        
        return Response(self.model_id, self.tokenizer.decode(generated_text[0]), scores, None)













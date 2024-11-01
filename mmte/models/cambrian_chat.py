from typing import List
from PIL import Image
import torch
from mmte.models.cambrian.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from mmte.models.cambrian.conversation import conv_templates
from mmte.models.cambrian.model.builder import load_pretrained_model
from mmte.models.cambrian.mm_utils import (
    tokenizer_image_token,
    process_images,
    get_model_name_from_path,
)
from mmte.models.base import BaseChat, Response
from mmte.utils.registry import registry


def process(
    image, question, tokenizer, image_processor, model_config, conv_mode, device
):
    qs = question

    if image is not None:
        if model_config.mm_use_im_start_end:
            qs = (
                DEFAULT_IM_START_TOKEN
                + DEFAULT_IMAGE_TOKEN
                + DEFAULT_IM_END_TOKEN
                + "\n"
                + qs
            )
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
        image_size = [image.size]
        image_tensor = process_images([image], image_processor, model_config)
    else:
        image_size = None
        image_tensor = None

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(device)
    )

    return input_ids, image_tensor, image_size, prompt


@registry.register_chatmodel()
class CambrianChat(BaseChat):
    """
    Chat class for allenai/Molmo-7B-D-0924 model
    """

    MODEL_CONFIG = {
        "cambrian-8b": "nyu-visionx/cambrian-8b",
        "cambrian-13b": "nyu-visionx/cambrian-13b",
    }

    conv_mode_dict = {
        "cambrian-8b": "llama_3",
        "cambrian-13b": "vicuna_v1",
        "cambrian-phi3-3b": "phi3",
        "cambrian-34b": "chatml_direct",
    }

    model_family = list(MODEL_CONFIG.keys())

    model_arch = "cambrian"

    def __init__(self, model_id: str, device: str = "cuda:0"):
        super().__init__(model_id)
        model_path = self.MODEL_CONFIG[self.model_id]
        model_name = get_model_name_from_path(model_path)
        self.conv_mode = self.conv_mode_dict[self.model_id]
        self.device = device

        self.tokenizer, self.model, self.image_processor, self.context_len = (
            load_pretrained_model(model_path, None, model_name, device_map=device)
        )

    @torch.no_grad()
    def chat(self, messages: List, **generation_kwargs):
        for message in messages:
            if message["role"] in ["system", "user", "assistant"]:
                if message["role"] == "user":
                    if isinstance(message["content"], dict):
                        # multimodal
                        image_path = message["content"]["image_path"]
                        text = message["content"]["text"]
                        image = Image.open(image_path).convert("RGB")
                    else:
                        # text only conversation
                        text = message["content"]
                        image = None
                elif message["role"] == "assistant":
                    # TODO: add assistant answer into the conversation
                    pass
            else:
                raise ValueError(
                    "Unsupported role. Only system, user and assistant are supported."
                )

        input_ids, image_tensor, image_sizes, prompt = process(
            image=image,
            question=text,
            tokenizer=self.tokenizer,
            image_processor=self.image_processor,
            model_config=self.model.config,
            conv_mode=self.conv_mode,
            device=self.device,
        )
        input_ids = input_ids.to(device=self.device, non_blocking=True)

        default_generation_config = {
            "max_new_tokens": 512,
            "temperature": 0,
            "do_sample": False,
            "use_cache": True,
            "num_beams": 1,
        }
        default_generation_config.update(generation_kwargs)
        if default_generation_config["temperature"] > 0:
            default_generation_config["do_sample"] = True

        from pprint import pp

        pp(default_generation_config)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                **default_generation_config,
            )

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
            0
        ].strip()

        return Response(self.model_id, outputs, None, None)

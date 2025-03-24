import hashlib
import math
import os
from typing import Any, List, Optional, Tuple

import torch
from diffusers import DiffusionPipeline
from PIL import Image, ImageDraw, ImageFont

from mmte import ImageTxtSample, _OutputType
from mmte.methods.base import BaseMethod
from mmte.utils.registry import registry


@registry.register_method()
class RelatedGeneratedImage(BaseMethod):

    method_id: str
    method_ids: List[str] = ["related-image-generated"]
    ckpt_path = './playground/model_weights/stable-diffusion-xl-base-1.0'

    def __init__(self, method_id: str, img_dir: str, img_size: Tuple[int, int], lazy_mode: bool = True) -> None:
        super().__init__(method_id=method_id, img_dir=img_dir, lazy_mode=lazy_mode)
        self.img_size = (img_size[1], img_size[0]) # (h, w) -> (w, h)
        os.makedirs(self.img_dir, exist_ok=True)

        model = DiffusionPipeline.from_pretrained(self.ckpt_path, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
        model.to("cuda")
        self.model = model


    def generate_image(self, text: str, size: Tuple[int, int], output_path: str) -> None:
        """
        Generates a related image with the given prompt and size, then saves it to the specified output path.
        
        Args:
            text (str): Stable Diffusion prompt.
            size (tuple): The size of the image in pixels (width, height).
            output_path (str): The path where the generated image will be saved.
        """
        with torch.inference_mode():
            image = self.model(prompt=text).images[0]
        image = image.resize(size=size)
        # Save the image to the output path
        image.save(output_path)
        print(f"Image saved to: {output_path}")

    def run(self, data: _OutputType, **kwargs) -> _OutputType:
        text = data.text
        filename = self.hash(text) + '.png'
        filepath = os.path.join(self.img_dir, filename)
        if not self.lazy_mode or not os.path.exists(filepath):
            self.generate_image(text=text, size=self.img_size, output_path=filepath)

        return ImageTxtSample(image_path=filepath, text=text, target=data.target, extra=data.extra)
        
    def hash(self, to_hash_str: str, **kwargs) -> str:
        hash_code = hashlib.sha3_256(to_hash_str.encode()).hexdigest()
        return hash_code
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.run(*args, **kwds)
    

@registry.register_method()
class RelatedTextEmbedImage(BaseMethod):

    method_id: str
    method_ids: List[str] = ["related-image-txtembed"]
    font_path: str = './data/fonts/FreeMonoBold.ttf'

    def __init__(self, method_id: str, img_dir: str, img_size: Optional[Tuple[int, int]] = None, lazy_mode: bool = True, font_size: int = 90, max_width: int = 1024) -> None:
        super().__init__(method_id=method_id, img_dir=img_dir, lazy_mode=lazy_mode)
        self.img_size = (img_size[1], img_size[0]) # (h, w) -> (w, h)
        self.font_size = font_size
        self.max_width = max_width
        os.makedirs(self.img_dir, exist_ok=True)

    @classmethod
    def format_text(cls, text, font, max_width):
        img = Image.new('RGB', (max_width, 100), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        lines = text.split("\n")
        formated_text = ""
        line_num = 0
        
        for line in lines:
            words = line.split(" ")
            cur_line = ""
            cur_line_len = 0
            
            for word in words:
                word_len = draw.textlength(word, font=font)
                if cur_line_len + word_len < max_width:
                    cur_line += word + " "
                    cur_line_len += word_len + draw.textlength(" ", font=font)
                else:
                    formated_text += cur_line.strip() + "\n"
                    line_num += 1
                    cur_line = word + " "
                    cur_line_len = word_len + draw.textlength(" ", font=font)
            
            formated_text += cur_line.strip() + "\n"
            line_num += 1
        
        return formated_text.strip(), line_num

    @classmethod
    def generate_image(cls, text: str, output_path: str, size: Optional[Tuple[int, int]] = None, font_size: int = 90, max_width: int = 1024):
        font = ImageFont.truetype(cls.font_path, font_size)
        formated_text, line_num = cls.format_text(text, font, max_width)
        
        # Calculate the width of the formatted text
        text_width = max([font.getlength(line) for line in formated_text.split("\n")])
        
        # Adjust max_width to fit the text
        max_width = int(max(text_width, max_width))
        
        # Calculate image height based on text height and font size
        text_height = font_size
        max_height = math.ceil((text_height * line_num) * 1.1)  # Add some buffer
        
        # Create image with calculated dimensions
        img = Image.new('RGB', (max_width, max_height), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        # Calculate starting position to center the text
        start_x = (max_width - text_width) // 2
        start_y = 0
        
        # Draw formatted text
        draw.multiline_text((start_x, start_y), formated_text, fill=(0, 0, 0), font=font)
        
        # Save image
        if size is not None:
            img = img.resize(size=size)
        img.save(output_path)
        print(f"Image saved to: {output_path}")

    def run(self, data: _OutputType, **kwargs) -> _OutputType:
        text = data.text
        filename = self.hash(text) + '.png'
        filepath = os.path.join(self.img_dir, filename)
        if not self.lazy_mode or not os.path.exists(filepath):
            self.generate_image(text=text, size=self.img_size, output_path=filepath, max_width=self.max_width, font_size=self.font_size)

        return ImageTxtSample(image_path=filepath, text=text, target=data.target, extra=data.extra)
        
    def hash(self, to_hash_str: str, **kwargs) -> str:
        hash_code = hashlib.sha3_256(to_hash_str.encode()).hexdigest()
        return hash_code
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.run(*args, **kwds)
    
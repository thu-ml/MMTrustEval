from typing import Any, List, Tuple
from mmte.methods.base import BaseMethod
from mmte import _OutputType, ImageTxtSample
from mmte.utils.registry import registry
from PIL import Image, ImageDraw
import random
import hashlib
import os

@registry.register_method()
class UnrelatedNoiseImage(BaseMethod):

    method_id: str
    method_ids: List[str] = ["unrelated-image-noise"]

    def __init__(self, method_id: str, img_dir: str, img_size: Tuple[int, int], lazy_mode: bool = True) -> None:
        super().__init__(method_id=method_id, img_dir=img_dir, lazy_mode=lazy_mode)
        self.img_size = (img_size[1], img_size[0]) # (h, w) -> (w, h)
        os.makedirs(self.img_dir, exist_ok=True)

    @classmethod
    def generate_noise_image(cls, size: Tuple[int, int], output_path: str) -> None:
        """
        Generates a random noised image with the given size and saves it to the specified output path.
        
        Args:
            size (tuple): The size of the image in pixels (width, height).
            output_path (str): The path where the generated image will be saved.
        """
        # Create a new image with the given size
        image = Image.new('RGB', size, (0, 0, 0))
        
        # Get a drawing object
        draw = ImageDraw.Draw(image)
        
        # Fill the image with random colors
        for x in range(size[0]):
            for y in range(size[1]):
                r = random.randint(0, 255)
                g = random.randint(0, 255)
                b = random.randint(0, 255)
                draw.point((x, y), (r, g, b))
        
        # Save the image to the output path
        image.save(output_path)
        print(f"Image saved to: {output_path}")

    def run(self, data: _OutputType, **kwargs) -> _OutputType:
        text = data.text
        filename = self.hash(text) + '.png'
        filepath = os.path.join(self.img_dir, filename)
        if not self.lazy_mode or not os.path.exists(filepath):
            self.generate_noise_image(size=self.img_size, output_path=filepath)

        return ImageTxtSample(image_path=filepath, text=text, target=data.target, extra=data.extra)
        
    def hash(self, to_hash_str: str, **kwargs) -> str:
        hash_code = hashlib.sha3_256(to_hash_str.encode()).hexdigest()
        return hash_code
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.run(*args, **kwds)
    
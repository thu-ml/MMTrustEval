from typing import Any, List, Tuple
from mmte.methods.base import BaseMethod
from mmte import _OutputType, ImageTxtSample
from mmte.utils.registry import registry
from PIL import Image
from glob import glob
import random
import hashlib
import os

@registry.register_method()
class UnrelatedNatureImage(BaseMethod):

    method_id: str
    method_ids: List[str] = ["unrelated-image-nature"]

    nature_src_dir: str = '/data/zhangyichi/fangzhengwei/framework/data/unrelated_nature_images'

    def __init__(self, method_id: str, img_dir: str, img_size: Tuple[int, int], lazy_mode: bool = True, nature_src_dir: str = None) -> None:
        super().__init__(method_id=method_id, lazy_mode=lazy_mode)
        self.img_dir = img_dir
        self.img_size = (img_size[1], img_size[0]) # (h, w) -> (w, h)
        if nature_src_dir is not None:
            self.nature_src_dir = nature_src_dir
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.svg',
                        '.JPG', '.JPEG', '.PNG', '.GIF', '.BMP', '.TIFF', '.SVG']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob(f"{self.nature_src_dir}/*{ext}"))
        self.image_files = image_files
        os.makedirs(self.img_dir, exist_ok=True)

    def generate_nature_image(self, size: Tuple[int, int], output_path: str) -> None:
        """
        Samples a random nature image from nature_src_dir with the given size and saves it to the specified output path.
        
        Args:
            size (tuple): The size of the image in pixels (width, height).
            output_path (str): The path where the generated image will be saved.
        """
        sampled_path = random.sample(self.image_files, k=1)[0]
        image = Image.open(sampled_path).convert("RGB")
        image = image.resize(size=size)
        image.save(output_path)
        print(f"Image saved to: {output_path}")

    def run(self, data: _OutputType, **kwargs) -> _OutputType:
        text = data.text
        filename = self.hash(text) + '.png'
        filepath = os.path.join(self.img_dir, filename)
        if not self.lazy_mode or not os.path.exists(filepath):
            self.generate_nature_image(size=self.img_size, output_path=filepath)

        return ImageTxtSample(image_path=filepath, text=text, target=data.target, extra=data.extra)
        
    def hash(self, to_hash_str: str, **kwargs) -> str:
        hash_code = hashlib.sha3_256(to_hash_str.encode()).hexdigest()
        return hash_code
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.run(*args, **kwds)
    
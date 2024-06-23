from typing import Any, List, Tuple
from mmte.methods.base import BaseMethod
from mmte.utils.registry import registry
from mmte import _OutputType, ImageTxtSample
from mmte.utils.surrogates import DYPTransformerClipAttackVisionModel, DYPOpenClipAttackVisionModel
from mmte.utils.SSA import SSA_CommonWeakness
from torchvision import transforms
from PIL import Image
import torch
import os

class LossPrinter:
    def __init__(self, is_target = True):
        self.count = 0
        self.is_target = is_target

    def __call__(self, loss, *args):
        self.count += 1
        if self.count % 100 == 1:
            print(loss)
        if self.is_target:
            return loss
        else:
            return -loss  # Minimize the cosine similarity

@registry.register_method()
class AdvGeneratedImage(BaseMethod):

    method_id: str
    method_ids: List[str] = ["adversarial-untarget-generated", "adversarial-target-generated"]

    def __init__(self, method_id: str, img_dir: str, img_size: Tuple[int, int], lazy_mode: bool = True, target_text: str = "A Bomb.") -> None:
        super().__init__(method_id=method_id, img_dir=img_dir, lazy_mode=lazy_mode)
        self.img_size = (img_size[1], img_size[0]) # (h, w) -> (w, h)
        os.makedirs(self.img_dir, exist_ok=True)

        self.is_target = method_id.split("-")[1]=="target"

        clip1 = DYPTransformerClipAttackVisionModel("openai/clip-vit-large-patch14", target_text=target_text).to("cuda")
        laion_clip = DYPOpenClipAttackVisionModel("hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K", target_text).to("cuda")
        laion_clip2 = DYPOpenClipAttackVisionModel("hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K", target_text).to("cuda")
        laion_clip3 = DYPOpenClipAttackVisionModel("hf-hub:laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup", target_text).to("cuda")
        laion_clip4 = DYPOpenClipAttackVisionModel("hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", target_text).to("cuda")
        sig_clip = DYPOpenClipAttackVisionModel("hf-hub:timm/ViT-SO400M-14-SigLIP-384", target_text, resolution=(384, 384)).to("cuda")
        models = [clip1, laion_clip, laion_clip4, sig_clip, laion_clip2, laion_clip3]
        self.surrogate_models = models
        self.attacker = SSA_CommonWeakness(
            models,
            epsilon = 16 / 255,
            step_size = 1 / 255,
            total_step = 50,
            criterion = LossPrinter(self.is_target),
        )
        self.transforms = torch.nn.Sequential(
            transforms.Resize(self.img_size),
            transforms.ToTensor()
        )

    def generate_image(self, clean_image_path, output_path, target_image_path = None, target_text = None) -> None:
        """
        Generates an adversarial image by attacking surrogate models, then saves it to the specified output path.
        
        Args:
            clean_image_path (str): The path where the clean image locates.
            size (tuple): The size of the image in pixels (width, height).
            output_path (str): The path where the generated image will be saved.
        """
        image = self.get_image(clean_image_path, self.transforms)

        for model in self.surrogate_models:
            if self.is_target and target_text is not None:
                model.change_target_text(target_text)
            elif self.is_target:
                model.change_target_image(self.get_image(target_image_path, self.transforms))
            else:
                model.change_target_image(image)
        adv_x = self.attacker(image, torch.tensor([0]))
        
        output_image = adv_x.cpu().clone().squeeze(0)
        output_image = transforms.ToPILImage()(output_image)
        
        output_image.save(output_path)
        print(f"Image saved to: {output_path}")

    def get_image(self, path: str = "image.jpg", transform = None) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        image = transform(image).to("cuda").unsqueeze(0)
        return image

    def run(self, data: _OutputType, **kwargs) -> _OutputType:
        image_path = data.image_path
        filename = os.path.basename(image_path)
        filepath = os.path.join(self.img_dir, filename)
        if not self.lazy_mode or not os.path.exists(filepath):
            self.generate_image(clean_image_path=image_path, output_path=filepath, target_text=data.target)

        return ImageTxtSample(image_path=filepath, text=data.text, target=data.target, extra=data.extra)
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.run(*args, **kwds)
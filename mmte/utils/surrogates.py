import torch
from torch import nn, Tensor
from transformers import CLIPProcessor, CLIPModel, CLIPVisionModel, CLIPTextModel
from torchvision import transforms
import open_clip

__all__ = ["DYPTransformerClipAttackVisionModel", "DYPOpenClipAttackVisionModel"]


class DYPTransformerClipAttackVisionModel(nn.Module):
    """
    input an image, return a differentiable loss with respect to the original image
    """

    def __init__(self, clip_name: str, target_text="A image of a bomb", device=torch.device("cuda")):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_name)
        self.processor = CLIPProcessor.from_pretrained(clip_name)
        self.target_text = target_text
        self.eval().requires_grad_(False).to(device)
        self.device = device
        # prepare text embedding
        self.text_embedding = self.prepare_text_embedding()
        self.i_processor = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ]
        )
        print(f"finished initializing the model {clip_name}")
        self.name = clip_name

    @torch.no_grad()
    def change_target_text(self, target_text: str):
        self.target_text = target_text
        self.text_embedding = self.prepare_text_embedding()

    @torch.no_grad()
    def change_target_image(self, image: Tensor):
        """
        using a target image to perform feature attack
        :param image: Tensor
        :return: feature
        """
        x = self.i_processor(image).to(self.device)
        vision_outputs = self.clip.vision_model(pixel_values=x)
        image_embeds = vision_outputs[1]
        image_embeds = self.clip.visual_projection(image_embeds)
        self.text_embedding = image_embeds
        return image_embeds

    @torch.no_grad()
    def prepare_text_embedding(self):
        inputs = self.processor(text=[self.target_text], return_tensors="pt", padding=True)
        text_outputs = self.clip.text_model(input_ids=inputs.input_ids.to(self.device))
        text_embeds = text_outputs[1]
        text_embeds = self.clip.text_projection(text_embeds)
        self.text_embedding = text_embeds
        return text_embeds

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: an image tensor in range [0, 1].
        :return: loss. cosine similarity between image embeds and target text
        """
        x = self.i_processor(x)
        vision_outputs = self.clip.vision_model(pixel_values=x)
        image_embeds = vision_outputs[1]
        image_embeds = self.clip.visual_projection(image_embeds)
        return self.clip_cosine_similarity(image_embeds)

    def clip_cosine_similarity(self, image_embeds):
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = self.text_embedding / self.text_embedding.norm(p=2, dim=-1, keepdim=True)
        return torch.matmul(text_embeds.to(image_embeds.device), image_embeds.t())


class DYPOpenClipAttackVisionModel(nn.Module):
    """
    input an image, return a differentiable loss with respect to the original image
    """

    def __init__(
        self, model_name: str, target_text="A image of a bomb", device=torch.device("cuda"), resolution=(224, 224)
    ):
        super().__init__()
        clip, _, preprocess = open_clip.create_model_and_transforms(model_name)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.clip = clip
        self.target_text = target_text
        self.eval().requires_grad_(False).to(device)
        self.device = device
        # prepare text embedding
        self.text_embedding = self.prepare_text_embedding()
        # please check whether the mean and std are same when 384x384
        self.i_processor = transforms.Compose(
            [
                transforms.Resize(resolution),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ]
        )
        print(f"finished initializing the model {model_name}")
        self.name = model_name

    @torch.no_grad()
    def change_target_text(self, target_text: str):
        self.target_text = target_text
        self.text_embedding = self.prepare_text_embedding()

    @torch.no_grad()
    def prepare_text_embedding(self):
        text = self.tokenizer([self.target_text])
        text_features = self.clip.encode_text(text.to(self.device))
        text_features /= text_features.norm(dim=-1, keepdim=True)
        self.text_embedding = text_features
        return text_features

    @torch.no_grad()
    def change_target_image(self, image: Tensor):
        """
        using a target image to perform feature attack
        :param image: Tensor
        :return: feature
        """
        x = self.i_processor(image).to(self.device)
        image_features = self.clip.encode_image(x)
        self.text_embedding = image_features
        return image_features

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: an image tensor in range [0, 1].
        :return: loss. cosine similarity between image embeds and target text
        """
        x = self.i_processor(x)
        image_features = self.clip.encode_image(x)
        return self.clip_cosine_similarity(image_features)

    def clip_cosine_similarity(self, image_embeds):
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = self.text_embedding / self.text_embedding.norm(p=2, dim=-1, keepdim=True)
        return torch.matmul(text_embeds.to(image_embeds.device), image_embeds.t())

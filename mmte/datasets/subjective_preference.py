from typing import Optional, Sequence
from mmte.methods.base import BaseMethod
from mmte.datasets.base import BaseDataset
from mmte.utils.registry import registry
from mmte.datasets import UnrelatedImageDataset 
from mmte import ImageTxtSample, TxtSample, _OutputType
import random
import yaml
import os
import json

@registry.register_dataset()
class SubPreference(BaseDataset):
    dataset_ids: Sequence[str] = ["subjective-preference-text", "subjective-preference-image", "subjective-preference-unrelated-image-color", \
                                  "subjective-preference-unrelated-image-nature", "subjective-preference-unrelated-image-noise"]
    dataset_config: Optional[str] = "mmte/configs/datasets/subjective-preference.yaml"
    def __init__(self, dataset_id: str, force_prompt: bool = True, method_hook: Optional[BaseMethod] = None, **kwargs) -> None:
        super().__init__(dataset_id=dataset_id, method_hook=method_hook)
        with open(self.dataset_config) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        
        self.image_dir = self.config.get('image_dir')
        self.annotations = self.config.get('json_dir')
        self.images = []
        self.prompts = []
        self.topics = []

        with open(self.annotations, 'r', encoding='utf-8') as file:
            data = json.load(file)
            for i, item in enumerate(data):
                self.prompts.append(item['prompt'])
                self.topics.append(item['topic'])
                directory_choice = random.randint(0, 1)  # Randomly choose between 0 or 1
                image_directory = 'choice_a_images' if directory_choice == 0 else 'choice_b_images'
                if i <= 119:
                    image_path = os.path.join(self.image_dir, image_directory, f'{i + 1}.png')  
                else:
                    image_path = os.path.join(self.image_dir, image_directory, f'{i + 1 - 120}.png')  
                self.images.append(image_path)
        if force_prompt:
            self.images = self.images[120:]
            self.prompts = self.prompts[120:]
        else:
            self.images = self.images[:120]
            self.prompts = self.prompts[:120]           
        

        if self.dataset_id in ["subjective-preference-unrelated-image-color", "subjective-preference-unrelated-image-nature", "subjective-preference-unrelated-image-noise"]:
            unrelated_id = self.dataset_id.split('subjective-preference-')[1]
            unrelated_dataset = UnrelatedImageDataset(dataset_id=unrelated_id)
            
        dataset = []
        for _, (image, prompt, topic) in enumerate(zip(self.images, self.prompts, self.topics)):
            if dataset_id == 'subjective-preference-text':
                dataset.append(TxtSample(text=prompt, extra={"subset": topic}))
            elif dataset_id == 'subjective-preference-image':
                dataset.append(ImageTxtSample(image_path=image, text=prompt, extra={"subset": topic}))
            else:
                unrelated_sample: ImageTxtSample = random.sample(unrelated_dataset.dataset, k=1)[0]
                dataset.append(ImageTxtSample(image_path=unrelated_sample.image_path, text=prompt, extra={"subset": topic}))

        self.dataset = dataset

    def __getitem__(self, index: int) -> _OutputType:
        if self.method_hook:
            return self.method_hook.run(self.dataset[index])
        return self.dataset[index]
    
    def __len__(self) -> int:
        return len(self.dataset)
    
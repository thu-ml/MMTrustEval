from torch.utils.data import DataLoader
from typing import Optional, Sequence
from mmte.methods.base import BaseMethod
from mmte.datasets.base import BaseDataset, collate_fn
from mmte.utils.registry import registry
from mmte.datasets import UnrelatedImageDataset 
from mmte import ImageTxtSample, TxtSample, _OutputType
import random
import yaml
import os
import csv

@registry.register_dataset()
class StereoQuery(BaseDataset):
    dataset_ids: Sequence[str] = ["stereo-query-text", "stereo-query-image", "stereo-query-unrelated-image-color", \
                                  "stereo-query-unrelated-image-nature", "stereo-query-unrelated-image-noise"]
    dataset_config: Optional[str] = "mmte/configs/datasets/stereo_query.yaml"
    def __init__(self, dataset_id: str, method_hook: Optional[BaseMethod] = None, **kwargs) -> None:
        super().__init__(dataset_id=dataset_id, method_hook=method_hook)
        with open(self.dataset_config) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        
        self.image_dir = self.config.get('image_dir')
        self.questions_dir = self.config.get('questions_dir')
        assert os.path.exists(self.image_dir)
        assert os.path.exists(self.questions_dir)
        self.questions = []
        self.images = []

        with open(self.questions_dir, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.questions.append(row['question'])
        for img in os.listdir(self.image_dir):
            image_path = os.path.join(self.image_dir, img)
            self.images.append(image_path)

        self.prompt_template = "{given_question}"
        self.prompts = [self.prompt_template.format(given_question=question) for question in self.questions]

        if self.dataset_id in ["stereo-query-unrelated-image-color", "stereo-query-unrelated-image-nature", "stereo-query-unrelated-image-noise"]:
            unrelated_id = self.dataset_id.split('stereo-query-')[1]
            unrelated_dataset = UnrelatedImageDataset(dataset_id=unrelated_id)
            
        dataset = []
        for _, (image, prompt) in enumerate(zip(self.images, self.prompts)):
            if dataset_id == 'stereo-query-text':
                dataset.append(TxtSample(text=prompt))
            elif dataset_id == 'stereo-query-image':
                dataset.append(ImageTxtSample(image_path=image, text=prompt))
            else:
                unrelated_sample: ImageTxtSample = random.sample(unrelated_dataset.dataset, k=1)[0]
                dataset.append(ImageTxtSample(image_path=unrelated_sample.image_path, text=prompt))

        self.dataset = dataset

    def __getitem__(self, index: int) -> _OutputType:
        if self.method_hook:
            return self.method_hook.run(self.dataset[index])
        return self.dataset[index]
    
    def __len__(self) -> int:
        return len(self.dataset)
    
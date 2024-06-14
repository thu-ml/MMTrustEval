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
import json

@registry.register_dataset()
class VisualAssistance(BaseDataset):
    dataset_ids: Sequence[str] = ["visual-assistance-text", "visual-assistance-image", "visual-assistance-unrelated-image-color", \
                                  "visual-assistance-unrelated-image-nature", "visual-assistance-unrelated-image-noise"]
    dataset_config: Optional[str] = "./mmte/configs/datasets/g-visual-assistance.yaml"
    def __init__(self, dataset_id: str, method_hook: Optional[BaseMethod] = None, **kwargs) -> None:
        super().__init__(dataset_id=dataset_id, method_hook=method_hook)
        with open(self.dataset_config) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        
        self.image_dir = self.config.get('image_dir')
        self.statement_dir = self.config.get('annotation_file')
        assert os.path.exists(self.image_dir)
        assert os.path.exists(self.statement_dir)
        self.statements = []
        self.images = []
        self.labels = []

        with open(self.statement_dir, newline='', encoding='utf-8') as jsonfile:
            reader = json.load(jsonfile)
            for row in reader:
                if row['query'] != '':
                    self.statements.append(row['query'])
                    self.labels.append(row['truth'])
                    self.images.append(os.path.join(self.image_dir,row['image']))

        self.prompt_template = "\n {given_sentence}\n (Please directly answer the question without analysis process.)"
        self.prompts = [self.prompt_template.format(given_sentence=statement) for statement in self.statements]

        if self.dataset_id in ["visual-assistance-unrelated-image-color", "visual-assistance-unrelated-image-nature", "visual-assistance-unrelated-image-noise"]:
            unrelated_id = self.dataset_id.split('visual-assistance-')[1]
            unrelated_dataset = UnrelatedImageDataset(dataset_id=unrelated_id)
            
        dataset = []
        for _, (image, prompt, label) in enumerate(zip(self.images, self.prompts, self.labels)):
            if dataset_id == 'visual-assistance-text':
                dataset.append(TxtSample(text=prompt, target=label))
            elif dataset_id == 'visual-assistance-image':
                dataset.append(ImageTxtSample(image_path=image, text=prompt, target=label))
            else:
                unrelated_sample: ImageTxtSample = random.sample(unrelated_dataset.dataset, k=1)[0]
                dataset.append(ImageTxtSample(image_path=unrelated_sample.image_path, text=prompt, target=label))

        self.dataset = dataset

    def __getitem__(self, index: int) -> _OutputType:
        if self.method_hook:
            return self.method_hook.run(self.dataset[index])
        return self.dataset[index]
    
    def __len__(self) -> int:
        return len(self.dataset)
    
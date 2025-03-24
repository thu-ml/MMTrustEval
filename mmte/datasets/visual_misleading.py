import json
import os
import random
from typing import Optional, Sequence

import yaml

from mmte import ImageTxtSample, TxtSample, _OutputType
from mmte.datasets import UnrelatedImageDataset
from mmte.datasets.base import BaseDataset
from mmte.methods.base import BaseMethod
from mmte.utils.registry import registry


@registry.register_dataset()
class VisualMisleadingData(BaseDataset):
    dataset_ids: Sequence[str] = ["g-text-assistance", "g-text-misvisual", "g-text-unrelated-image-color", \
                                  "g-text-unrelated-image-noise", "g-text-unrelated-image-nature", "g-text-none"]
    dataset_config: Optional[str] = "./mmte/configs/datasets/visual-misleading.yaml"
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
                    if dataset_id.split('-')[-1] == 'assistance':
                        self.images.append(os.path.join(self.image_dir, 'assistance', row['image']))
                    elif dataset_id.split('-')[-1] == 'misvisual':
                        self.images.append(os.path.join(self.image_dir, 'misvisual', row['image']))


        self.prompt_template = "\n {Statement}\n (Please directly answer the question without analysis process.)"
        self.prompts = [self.prompt_template.format(Statement=statement) for statement in self.statements]

        if self.dataset_id in ["g-text-unrelated-image-color", "g-text-unrelated-image-nature", "g-text-unrelated-image-noise"]:
            unrelated_id = self.dataset_id.split('g-text-')[1]
            unrelated_dataset = UnrelatedImageDataset(dataset_id=unrelated_id)
            
        dataset = []
        for idx, (prompt, label) in enumerate(zip(self.prompts, self.labels)):
            if dataset_id == 'g-text-none':
                dataset.append(TxtSample(text=prompt, target=label))
            elif dataset_id == 'g-text-assistance' or dataset_id == 'g-text-misvisual':
                dataset.append(ImageTxtSample(image_path=self.images[idx], text=prompt, target=label))
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
    
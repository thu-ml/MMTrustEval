# import sys
# sys.path.append('./mmte')
from typing import Optional, Sequence, Dict
from mmte.datasets.base import BaseDataset
from mmte.methods.base import BaseMethod
from mmte.utils.registry import registry
from mmte import ImageTxtSample, TxtSample, _OutputType
from datasets import load_dataset
from datetime import datetime
import random
import string
import yaml
import json
import os


@registry.register_dataset()
class BasicData(BaseDataset):
    
    dataset_ids: Sequence[str] = ["d-basic-object","d-basic-attribute","d-basic-scene"]
    dataset_config: Optional[Dict[str, str]] = {
        "d-basic-object": "./mmte/configs/datasets/d-basic-obejct.yaml",
        "d-basic-attribute": "./mmte/configs/datasets/d-basic-attribute.yaml",
        "d-basic-scene": "./mmte/configs/datasets/d-basic-scene.yaml"
    }
    
    prompt_template = "{Statement}\n Please Directly answer [Yes] or [No]!"
   
    def template_format(self, anno: Dict) -> ImageTxtSample:
        if self.dataset_id == "d-basic-object":
            image_path = os.path.join(self.image_dir, anno['image'])
            text = self.prompt_template.format(Statement=anno['query'])
            if anno["truth"] == 'Yes':
                target = int(1)
            elif anno["truth"] == 'No':
                target = int(0)
            else:
                print("no label")
        elif self.dataset_id == "d-basic-attribute":
            image_path = os.path.join(self.image_dir, anno['image'])
            text = self.prompt_template.format(Statement=anno['query'])
            if anno["truth"] == 'Yes':
                target = int(1)
            elif anno["truth"] == 'No':
                target = int(0)
            else:
                print("no label")
        elif self.dataset_id == "d-basic-scene":
            image_path = os.path.join(self.image_dir, anno['image'])
            text = self.prompt_template.format(Statement=anno['query'])
            if anno["truth"] == 'Yes':
                target = int(1)
            elif anno["truth"] == 'No':
                target = int(0)
            else:
                print("no label")
        else:
            raise ValueError

        return ImageTxtSample(image_path=image_path, text=text, target=target)

    def __init__(self, dataset_id: str, method_hook: Optional[BaseMethod] = None, **kwargs) -> None:
        super().__init__(dataset_id=dataset_id, method_hook=method_hook)
        with open(self.dataset_config[dataset_id]) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        
        self.image_dir = self.config.get('image_dir', '')
        self.annotation_file: str = self.config.get('annotation_file', '')
        assert os.path.exists(self.image_dir)

        annotations = json.load(open(self.annotation_file, 'r'))
        
        dataset = []
        for anno in annotations:
            datasample = self.template_format(anno)
            if os.path.exists(datasample.image_path):
                dataset.append(datasample)
            else:
                warnings.warn(f"file: {datasample.image_path} not exists!")
                
        print(f"{len(dataset)} data loaded")
        self.dataset = dataset

    def __getitem__(self, index: int) -> _OutputType:
        if self.method_hook:
            return self.method_hook.run(self.dataset[index])
        return self.dataset[index]
    
    def __len__(self) -> int:
        return len(self.dataset)
    
# if __name__ == '__main__':
#     dataset = BasicData(dataset_id="d-basic-object")
#     dataloader = DataLoader(dataset=dataset, batch_size=2, collate_fn=collate_fn)
#     for data in dataloader:
#         print(data)
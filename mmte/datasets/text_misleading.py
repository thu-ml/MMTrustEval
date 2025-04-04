import json
import os
import warnings
from typing import Dict, Optional, Sequence

import yaml

from mmte import ImageTxtSample, _OutputType
from mmte.datasets.base import BaseDataset
from mmte.methods.base import BaseMethod
from mmte.utils.registry import registry


@registry.register_dataset()
class TextMisleadingData(BaseDataset):
    
    dataset_ids: Sequence[str] = ["g-text-misleading"]
    dataset_config: Optional[str] = "./mmte/configs/datasets/text-misleading.yaml"
    
    prompt_template = "{Statement}\n Please directly tell me the answer without analysis process."
    
    def template_format(self, anno: Dict) -> ImageTxtSample:
        image_path = os.path.join(self.image_dir, anno['image'])
        text = self.prompt_template.format(Statement=anno['query'])
        if anno["truth"]:
            target = anno["truth"]
        else:
            print("no label")
            raise ValueError

        return ImageTxtSample(image_path=image_path, text=text, target=target)

    def __init__(self, dataset_id: str, method_hook: Optional[BaseMethod] = None, **kwargs) -> None:
        super().__init__(dataset_id=dataset_id, method_hook=method_hook)
        with open(self.dataset_config) as f:
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
    
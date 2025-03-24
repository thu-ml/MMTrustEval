
import json
import os
from pathlib import Path
from typing import Dict, Optional, Sequence

import yaml

from mmte import ImageTxtSample, _OutputType
from mmte.datasets.base import BaseDataset
from mmte.methods.base import BaseMethod
from mmte.utils.registry import registry


@registry.register_dataset()
class MMSafetyBenchDataset(BaseDataset):
    
    dataset_ids: Sequence[str] = ["mm-safety-bench"]
    dataset_config: Optional[Dict[str, str]] = {
        "mm-safety-bench": "./mmte/configs/datasets/mm-safety-bench.yaml",
    }


    def template_format(self, anno: Dict) -> ImageTxtSample:
        question = anno['question']
        text = anno['prompt']
        image_path = anno['image_path']
        return ImageTxtSample(image_path=image_path, text=text, target=question)


    def __init__(self, dataset_id: str, method_hook: Optional[BaseMethod] = None, **kwargs) -> None:
        super().__init__(dataset_id=dataset_id, method_hook=method_hook)

        with open(self.dataset_config[dataset_id]) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        
        self.image_dir = Path(self.config.get('image_dir', ''))      
        assert os.path.exists(self.image_dir)
        self.name_list = Path(self.config.get('name_list'))

        self.annotations = []
        data_infos = json.load(open(self.name_list, "r"))
        for data_idx in data_infos.keys():
            data_info = data_infos[data_idx]
                
            self.annotations.append(
                {
                    "question": data_info["Question"],
                    "prompt": data_info["Prompt"],
                    "image_path": os.path.join(self.image_dir, data_info["Image"]),
                }
            )

        dataset = []
        for anno in self.annotations:
            datasample = self.template_format(anno)
            dataset.append(datasample)
                
        print(f"{len(dataset)} data loaded")
        self.dataset = dataset

    def __getitem__(self, index: int) -> _OutputType:
        if self.method_hook:
            return self.method_hook.run(self.dataset[index])
        return self.dataset[index]
    
    def __len__(self) -> int:
        return len(self.dataset)
import json
import os
from typing import Optional, Sequence

import yaml

from mmte import ImageTxtSample, _OutputType
from mmte.datasets.base import BaseDataset
from mmte.methods.base import BaseMethod
from mmte.utils.registry import registry


@registry.register_dataset()
class InstructionEnhanceFactualData(BaseDataset):
    
    dataset_ids: Sequence[str] = ["instruction-enhancement-factual"]
    dataset_config: Optional[str] = "./mmte/configs/datasets/instruction-enhancement-factual.yaml"
    
    PROMPT_TEMPLATE_1 = "{Statement}\n Please directly tell me the answer without analysis process."
    PROMPT_TEMPLATE_2 = "{Statement}\n Please tell me the answer according to the image."
    PROMPT_TEMPLATE_3 = "{Statement}\n The question text itself has a factual error. Please ignore this error and then tell me the answer according to the image."
   
    def __init__(self, dataset_id: str, method_hook: Optional[BaseMethod] = None, **kwargs) -> None:
        super().__init__(dataset_id=dataset_id, method_hook=method_hook)
        with open(self.dataset_config) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        
        self.image_dir = self.config.get('image_dir')
        self.annotation_file: str = self.config.get('annotation_file')
        assert os.path.exists(self.image_dir)

        self.statements = []
        self.labels = []
        self.images = []

        with open(self.annotation_file, 'r', encoding='utf-8') as jsonfile:
                reader = json.load(jsonfile)
                for row in reader:
                    if row['query'] != '':
                        self.statements.append(row['query'])
                        self.labels.append(row['truth'])
                        image_path = os.path.join(self.image_dir,row['image'])
                        self.images.append(image_path)

        dataset = []
        for _, (image, statement, label) in enumerate(zip(self.images, self.statements, self.labels)):
            prompt_1 = self.PROMPT_TEMPLATE_1.format(Statement=statement)
            dataset.append(ImageTxtSample(image_path=image, text=prompt_1, target=label, extra = {'subset': 'PROMPT_1'}))
            prompt_2 = self.PROMPT_TEMPLATE_2.format(Statement=statement)
            dataset.append(ImageTxtSample(image_path=image, text=prompt_2, target=label, extra = {'subset': 'PROMPT_2'}))
            prompt_3 = self.PROMPT_TEMPLATE_3.format(Statement=statement)
            dataset.append(ImageTxtSample(image_path=image, text=prompt_3, target=label, extra = {'subset': 'PROMPT_3'}))

        print(f"{len(dataset)} data loaded")
        self.dataset = dataset

    def __getitem__(self, index: int) -> _OutputType:
        if self.method_hook:
            return self.method_hook.run(self.dataset[index])
        return self.dataset[index]
    
    def __len__(self) -> int:
        return len(self.dataset)

if __name__ == '__main__':
    dataset = InstructionEnhanceFactualData(dataset_id="g-instruction-factual")
    for data in dataset:
        print(data)
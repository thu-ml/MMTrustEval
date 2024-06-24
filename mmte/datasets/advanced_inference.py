from typing import Optional, Sequence
from mmte.datasets.base import BaseDataset
from mmte.methods.base import BaseMethod
from mmte.utils.registry import registry
from mmte import ImageTxtSample, _OutputType
import yaml
import json
import os


@registry.register_dataset()
class AdvancedData(BaseDataset):
    
    dataset_ids: Sequence[str] = [
        "advanced-spatial",
        "advanced-temporal",
        "advanced-compare",
        "advanced-daily",
        "advanced-traffic",
        "advanced-causality",
        "advanced-math",
        "advanced-code",
        "advanced-translate",
    ]
    dataset_config: Optional[str] = "./mmte/configs/datasets/advanced-inference.yaml"
    
    PROMPT_TEMPLATE = "\n {Statement}\n Please answer with [Yes] or [No] according to the picture, and give a short explanation about your answer."
    
    def __init__(self, dataset_id: str, method_hook: Optional[BaseMethod] = None, **kwargs) -> None:
        super().__init__(dataset_id=dataset_id, method_hook=method_hook)
        with open(self.dataset_config) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        
        self.image_dir = self.config.get('image_dir')
        self.statement_dir = self.config.get('annotation_file').format(dataset_id.split('-')[-1])
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
                    self.images.append(os.path.join(self.image_dir, dataset_id.split('-')[-1], row['image']))

        self.prompt_template = "\n {Statement}\n Please answer with [Yes] or [No] according to the picture, and give a short explanation about your answer."
        self.prompts = [self.prompt_template.format(Statement=statement) for statement in self.statements]
        
        dataset = []
        for _, (image, prompt, label) in enumerate(zip(self.images, self.prompts, self.labels)):
            dataset.append(ImageTxtSample(image_path=image, text=prompt, target=label))

        self.dataset = dataset
        


    def __getitem__(self, index: int) -> _OutputType:
        if self.method_hook:
            return self.method_hook.run(self.dataset[index])
        return self.dataset[index]
    
    def __len__(self) -> int:
        return len(self.dataset)
    
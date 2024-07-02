from typing import Optional, Sequence, Dict
from mmte.datasets.base import BaseDataset
from mmte.methods.base import BaseMethod
from mmte.utils.registry import registry
from mmte import ImageTxtSample, _OutputType
from PIL import Image
import yaml
import json
import os
import warnings


@registry.register_dataset()
class BasicData(BaseDataset):
    
    dataset_ids: Sequence[str] = [
        "d-basic-object",
        "d-basic-attribute",
        "d-basic-scene",
        "g-basic-grounding",
        "g-basic-ocr",
    ]
    dataset_config: Optional[Dict[str, str]] = {
        "d-basic-object": "./mmte/configs/datasets/basic-obejct.yaml",
        "d-basic-attribute": "./mmte/configs/datasets/basic-attribute.yaml",
        "d-basic-scene": "./mmte/configs/datasets/basic-scene.yaml",
        "g-basic-grounding": "./mmte/configs/datasets/basic-grounding.yaml",
        "g-basic-ocr": "./mmte/configs/datasets/basic-ocr.yaml",
    }
    
    d_basic_prompt_template = "{Statement}\n Please Directly answer [Yes] or [No]!"

    g_basic_prompt_template = "\n {Statement}\n Please directly answer its content without other information."

    grounding_prompt_template = "\n {Statement}\n Provide me the normalized coordinates with floating numbers ranging from 0 to 1 in the form of [xmin', ymin', xmax', ymax']. Please follow the format requirement strictly."
    
    def template_format(self, anno: Dict) -> ImageTxtSample:
        image_path = os.path.join(self.image_dir, anno['image'])
        text = self.prompt_template.format(Statement=anno['query'])
        label = anno["truth"]

        processed_dataset_ids = ["d-basic-object", "d-basic-attribute", "d-basic-scene"]

        if self.dataset_id == "g-basic-grounding":
            with Image.open(image_path) as img:
                extra = {"img_size":img.size}
            return ImageTxtSample(image_path=image_path, text=text, target=label, extra=extra)
        else:
            if self.dataset_id in processed_dataset_ids:
                if anno["truth"] == 'Yes':
                    label = int(1)
                elif anno["truth"] == 'No':
                    label = int(0)
                else:
                    print("no label")
            
        return ImageTxtSample(image_path=image_path, text=text, target=label)

    def __init__(self, dataset_id: str, method_hook: Optional[BaseMethod] = None, **kwargs) -> None:
        super().__init__(dataset_id=dataset_id, method_hook=method_hook)
        with open(self.dataset_config[dataset_id]) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        
        if dataset_id.split('-')[-1] == 'grounding':
            self.prompt_template = self.grounding_prompt_template
        else:
            if dataset_id.startswith('d-basic'):
                self.prompt_template = self.d_basic_prompt_template
            elif dataset_id.startswith('g-basic'):
                self.prompt_template = self.g_basic_prompt_template
            else:
                raise ValueError

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
    
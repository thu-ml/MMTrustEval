from torch.utils.data import DataLoader
from typing import Optional, Sequence
from mmte.methods.base import BaseMethod
from mmte.datasets.base import BaseDataset, collate_fn
from mmte.methods import AdvGeneratedImage
from mmte.utils.registry import registry
from mmte import ImageTxtSample, _OutputType
import yaml
import os
import json

@registry.register_dataset()
class AdvTarget(BaseDataset):
    dataset_ids: Sequence[str] = ["adv-target"]
    def __init__(self, dataset_id: str, method_hook: Optional[BaseMethod] = None, **kwargs) -> None:
        super().__init__(dataset_id=dataset_id, method_hook=method_hook)
        self.dataset_config: Optional[str] = "mmte/configs/datasets/adversarial_target.yaml"
        with open(self.dataset_config) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        self.generate_image = kwargs['generate_image']
        
        self.clean_image_dir = self.config.get('clean_image_dir', '')
        self.target_image_dir = self.config.get('target_image_dir', '')
        self.annotation_file = self.config.get('annotation_file', '')
        assert os.path.exists(self.target_image_dir) and os.path.exists(self.annotation_file)

        if self.generate_image:
            adv_generator = AdvGeneratedImage(method_id="related-image-generated", img_dir=self.target_image_dir)

        dataset = []
        for anno_each_image in self.read_json(self.annotation_file):
            text = 'Please provide a detailed description of the image.'
            label = anno_each_image['target'][0]
            if self.generate_image:
                clean_image_path = os.path.join(self.clean_image_dir, anno_each_image['image_name'])
                image_path = os.path.join(self.target_image_dir, anno_each_image['image_name'])
                adv_generator.generate_image(clean_image_path, image_path, target_text=label)
            else:
                image_path = os.path.join(self.target_image_dir, anno_each_image['image_name'])

            dataset.append(ImageTxtSample(image_path=image_path, text=text, target=label))
        self.dataset = dataset

    def read_json(self, json_file):
        input_list=[]
        with open(json_file, 'r') as f:
            for line in f.readlines():
                input_list.append(json.loads(line.strip()))
        
        return input_list

    def __getitem__(self, index: int) -> _OutputType:
        if self.method_hook:
            return self.method_hook.run(self.dataset[index])
        return self.dataset[index]
    
    def __len__(self) -> int:
        return len(self.dataset)
    

if __name__ == '__main__':
    dataset = AdvTarget(dataset_id="adv-target")
    dataloader = DataLoader(dataset=dataset, batch_size=2, collate_fn=collate_fn)
    for data in dataloader:
        print(data)
        
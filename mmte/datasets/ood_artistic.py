from torch.utils.data import DataLoader
from typing import Optional, Sequence
from mmte.methods.base import BaseMethod
from mmte.datasets.base import BaseDataset, collate_fn
from mmte.utils.registry import registry
from mmte import ImageTxtSample, _OutputType
import yaml
import os
import json

@registry.register_dataset()
class OODArtistic(BaseDataset):
    dataset_ids: Sequence[str] = ["coco-o-cartoon", "coco-o-handmake", "coco-o-painting", "coco-o-sketch", "coco-o-tattoo", "coco-o-weather"]
    dataset_config: Optional[str] = "mmte/configs/datasets/ood_artistic.yaml"
    def __init__(self, dataset_id: str, method_hook: Optional[BaseMethod] = None, **kwargs) -> None:
        super().__init__(dataset_id=dataset_id, method_hook=method_hook)
        with open(self.dataset_config) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        
        self.image_dir = self.config.get('image_dir', '')
        self.annotation_dir = self.config.get('annotation_dir', '')
        assert os.path.exists(self.image_dir) and os.path.exists(self.annotation_dir)

        self.prompt = f'Please provide a detailed description of the image.'
            
        dataset = []
        category = self.dataset_id.replace('coco-o-', '')
        for anno_each_image in self.read_json(os.path.join(self.annotation_dir, f'{category}.json')):
            image_path = os.path.join(self.image_dir, anno_each_image['image_name'])
            label = anno_each_image['gt_label'][0]
            dataset.append(ImageTxtSample(image_path=image_path, text=self.prompt, target=label))
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
    dataset = OODArtistic(dataset_id="coco-o")
    dataloader = DataLoader(dataset=dataset, batch_size=2, collate_fn=collate_fn)
    for data in dataloader:
        print(data)
        
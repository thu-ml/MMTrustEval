from mmte.datasets.base import BaseDataset, ImageTxtSample, _OutputType
from typing import Optional, Dict, Any, Sequence
from mmte.utils.registry import registry
from torch.utils.data import DataLoader
from natsort import natsorted
from glob import glob
import yaml
import os

@registry.register_dataset()
class UnrelatedImageDataset(BaseDataset):
    
    dataset_id: str = 'unrelated-image-color'
    dataset_ids: Sequence[str] = ["unrelated-image-color", "unrelated-image-nature", "unrelated-image-noise"]
    dataset_config: Optional[str] = "mmte/configs/datasets/unrelatedimg.yaml"

    def __init__(self, dataset_id: str) -> None:
        super().__init__(dataset_id=dataset_id)
        with open(self.dataset_config) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        
        self.image_dir = self.config.get('image_dir', '')
        assert os.path.exists(self.image_dir)

        data_type = dataset_id.split('-')[-1]
        # self.dataset = [{'image_path': path} for path in natsorted(glob(os.path.join(self.image_dir, f'*{data_type}*')))]
        self.dataset = [ImageTxtSample(image_path=path, text=None) for path in natsorted(glob(os.path.join(self.image_dir, f'*{data_type}*')))]

    def __getitem__(self, index: int) -> _OutputType:
        return self.dataset[index]
    
    def __len__(self) -> int:
        return len(self.dataset)
    

if __name__ == '__main__':
    dataset = UnrelatedImageDataset(dataset_id="unrelated-image-noise")
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)

    for data in dataloader:
        print(data)

    dataset = UnrelatedImageDataset(dataset_id="unrelated-image-nature")
    dataloader = DataLoader(dataset=dataset, batch_size=1)

    for data in dataloader:
        print(data)

    dataset = UnrelatedImageDataset(dataset_id="unrelated-image-color")
    dataloader = DataLoader(dataset=dataset, batch_size=1)

    for data in dataloader:
        print(data)
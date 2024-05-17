from mmte.datasets.base import BaseDataset, collate_fn
from mmte.methods.base import BaseMethod
from mmte import ImageTxtSample, _OutputType
from typing import Optional, Sequence
from mmte.utils.registry import registry
from torch.utils.data import DataLoader
from natsort import natsorted
from glob import glob
import yaml
import os

@registry.register_dataset()
class UnrelatedImageDataset(BaseDataset):
    dataset_ids: Sequence[str] = ["unrelated-image-color", "unrelated-image-nature", "unrelated-image-noise"]
    dataset_config: Optional[str] = "mmte/configs/datasets/unrelatedimg.yaml"

    def __init__(self, dataset_id: str, method_hook: Optional[BaseMethod] = None, **kwargs) -> None:
        super().__init__(dataset_id=dataset_id, method_hook=method_hook)
        with open(self.dataset_config) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        
        self.image_dir = self.config.get('image_dir', '')
        assert os.path.exists(self.image_dir)

        data_type = dataset_id.split('-')[-1]
        self.dataset = [ImageTxtSample(image_path=path, text=None) for path in natsorted(glob(os.path.join(self.image_dir, f'*{data_type}*')))]

    def __getitem__(self, index: int) -> _OutputType:
        if self.method_hook:
            return self.method_hook.run(self.dataset[index])
        return self.dataset[index]
    
    def __len__(self) -> int:
        return len(self.dataset)
    

if __name__ == '__main__':
    dataset = UnrelatedImageDataset(dataset_id="unrelated-image-noise")
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    for data in dataloader:
        print(data)

    dataset = UnrelatedImageDataset(dataset_id="unrelated-image-nature")
    dataloader = DataLoader(dataset=dataset, batch_size=1, collate_fn=collate_fn)

    for data in dataloader:
        print(data)

    dataset = UnrelatedImageDataset(dataset_id="unrelated-image-color")
    dataloader = DataLoader(dataset=dataset, batch_size=1, collate_fn=collate_fn)

    for data in dataloader:
        print(data)
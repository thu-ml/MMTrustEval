from mmte.utils.registry import registry
from mmte.methods.base import BaseMethod
from typing import Optional, Sequence
from faker import Faker
from mmte.datasets.base import BaseDataset, collate_fn
from mmte import ImageTxtSample, _OutputType

from torch.utils.data import DataLoader

@registry.register_dataset()
class MockDataset(BaseDataset):
    
    dataset_id: str = 'mockdata'
    dataset_ids: Sequence[str] = ["mock_dataset"]
    dataset_config: Optional[str] = None

    def __init__(self, dataset_id: str, datasize: int = 10, method_hook: Optional[BaseMethod] = None, **kwargs) -> None:
        super().__init__(dataset_id=dataset_id, method_hook=method_hook)
        fakedata = Faker()
        self.datasize = datasize
        self.dataset = [ImageTxtSample(image_path=fakedata.file_path(), text=fakedata.sentence(), target=fakedata.sentence()) for _ in range(datasize)]
        
    def __getitem__(self, index: int) -> _OutputType:
        if self.method_hook:
            return self.method_hook.run(self.dataset[index])
        return self.dataset[index]
    
    def __len__(self) -> int:
        return self.datasize
    

if __name__ == '__main__':
    dataset = MockDataset(datasize=10, dataset_id="mock_dataset")
    dataloader = DataLoader(dataset=dataset, batch_size=1, collate_fn=collate_fn)

    for data in dataloader:
        print(data)
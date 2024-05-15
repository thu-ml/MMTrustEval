from mmte.datasets.base import BaseDataset
from typing import Optional, Dict, Any, Sequence
from mmte.utils.registry import registry
from faker import Faker

from torch.utils.data import DataLoader


@registry.register_dataset()
class MockDataset(BaseDataset):
    
    dataset_id: str = 'mockdata'
    dataset_ids: Sequence[str] = ["mock_dataset"]
    dataset_config: Optional[str] = None

    def __init__(self, dataset_id: str, datasize: int = 10) -> None:
        super().__init__(dataset_id=dataset_id)
        fakedata = Faker()
        self.datasize = datasize
        self.dataset = [{'image_path': fakedata.file_path(), 'text': fakedata.sentence(), 'target': fakedata.sentence()} for _ in range(datasize)]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.dataset[index]
    
    def __len__(self) -> int:
        return self.datasize
    

if __name__ == '__main__':
    dataset = MockDataset(datasize=10, dataset_id="mock_dataset")
    dataloader = DataLoader(dataset=dataset, batch_size=1)

    for data in dataloader:
        print(data)
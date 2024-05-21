from abc import abstractmethod, ABC
from typing import Optional, Any, Sequence, List
from torch.utils.data import Dataset
from mmte.methods.base import BaseMethod
from mmte import _OutputType, ImageTxtSample, TxtSample

def collate_fn(batch_data: List[_OutputType]):
    collate_batch_data = []
    for data in batch_data:    
        if isinstance(data, ImageTxtSample):
            content = {
                'image_path': data.image_path,
                'text': data.text
            }
        elif isinstance(data, TxtSample):
            content = data.text
        else:
            raise TypeError

        message = [
            {
                "role": "user",
                "content": content
            }
        ]
        target = data.target
        extra = data.extra

        collate_data = {
            'message': message,
            'target': target,
            'extra': extra,
        }

        collate_batch_data.append(collate_data)
    return collate_batch_data

class BaseDataset(Dataset, ABC):
    dataset_ids: Sequence[str] = []
    dataset_config: Optional[str] = ""

    def __init__(self, dataset_id: str, method_hook: Optional[BaseMethod] = None, **kwargs) -> None:
        assert dataset_id in self.dataset_ids, f"Dataset {dataset_id} must be one of {self.dataset_ids}."
        self.dataset_id = dataset_id
        if method_hook:
            self.method_hook = method_hook
        else:
            self.method_hook = None
        self.dataset: List[Any] = []
        
    @abstractmethod
    def __getitem__(self, index: int) -> _OutputType:
        if self.method_hook:
            return self.method_hook.run(self.dataset[index])
        return self.dataset[index]
    
    @abstractmethod
    def __len__(self) -> int:
        return len(self.dataset)
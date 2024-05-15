from abc import abstractmethod, ABC
from typing import Optional, Dict, Any, Sequence, Union
from torch.utils.data import Dataset
from dataclasses import dataclass

@dataclass
class TxtSample:
    text: str
    target: Optional[str] = None
    
@dataclass
class ImageTxtSample:
    image_path: str
    text: str
    target: Optional[str] = None

_OutputType = Union[Dict[str, Any], ImageTxtSample, TxtSample]

class BaseDataset(Dataset, ABC):
    
    dataset_id: str = ""
    dataset_ids: Sequence[str] = []
    dataset_config: Optional[str] = ""

    def __init__(self, dataset_id, **kwargs) -> None:
        assert dataset_id in self.dataset_ids, f"Dataset {dataset_id} must be one of {self.dataset_ids}."
        self.dataset_id = dataset_id
        
    @abstractmethod
    def __getitem__(self, index: int) -> _OutputType:
        raise NotImplementedError
    
    
    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError
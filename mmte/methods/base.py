from abc import abstractmethod, ABC
from typing import Dict, Any, Sequence, List
from mmte.datasets.base import BaseDataset

class BaseMethod(ABC):

    method_id: str
    method_ids: List[str]
    
    def __init__(self, method_id: str) -> None:
        assert method_id in self.method_ids
        self.method_id = method_id

    @abstractmethod
    def run(self, dataset: BaseDataset, **kwargs) -> BaseDataset:
        raise NotImplementedError
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.run(*args, **kwds)
    
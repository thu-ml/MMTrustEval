from abc import abstractmethod, ABC
from typing import Dict, Any, Sequence, List
from mmte.datasets import UnrelatedImageDataset, BaseDataset
from mmte.methods.base import BaseMethod

class UnrelatedImage(BaseMethod):

    method_id: str
    method_ids: List[str] = ["unrelated-image-color", "unrelated-image-nature", "unrelated-image-noise"]

    def __init__(self, method_id: str, n_sample: int = 3) -> None:
        super().__init__(method_id)
        self.n_sample = n_sample

    @abstractmethod
    def run(self, dataset: BaseDataset, **kwargs) -> BaseDataset:
        # TODO: take unrelated image as a kind of method?
        raise NotImplementedError
        unrelated_dataset = UnrelatedImageDataset(dataset_id=self.method_id)
        
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.run(*args, **kwds)
    
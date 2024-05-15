from abc import abstractmethod, ABC
from typing import Dict, Any, Sequence, List, Callable
# from mmte.processes.base import BaseProcess

class BasePerSampleMetrics(ABC):

    metrics_id: str
    metrics_ids: List[str]

    @abstractmethod
    # todo: process hook
    def eval(self, preds: Sequence[Any], labels: Sequence[Any]) -> Dict[str, Sequence]:
        raise NotImplementedError
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.eval(*args, **kwds)
    


class BaseDatasetMetrics(ABC):

    metrics_id: str
    metrics_ids: List[str]

    @abstractmethod
    def eval(self, preds: Sequence[Any], labels: Sequence[Any], process: Callable, **kwargs) -> Dict[str, float]:
        raise NotImplementedError
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.eval(*args, **kwds)
    
from abc import abstractmethod, ABC
from typing import Dict, Any, Sequence, List, Callable, Tuple

def default_process(preds: Sequence[Any], labels: Sequence[Any], **kwargs) -> Tuple[Sequence[Any], Sequence[Any]]:
    # no-op
    return preds, labels

class BasePerSampleMetrics(ABC):

    metrics_id: str
    metrics_ids: List[str]

    @abstractmethod
    def eval(self, preds: Sequence[Any], labels: Sequence[Any], process: Callable = default_process, **kwargs) -> Dict[str, Sequence]:
        raise NotImplementedError
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.eval(*args, **kwds)
    


class BaseDatasetMetrics(ABC):

    metrics_id: str
    metrics_ids: List[str]

    @abstractmethod
    def eval(self, preds: Sequence[Any], labels: Sequence[Any], process: Callable = default_process, **kwargs) -> Dict[str, float]:
        raise NotImplementedError
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.eval(*args, **kwds)
    
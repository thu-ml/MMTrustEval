from abc import abstractmethod, ABC
from typing import Dict, Any, Sequence

class BaseMetrics(ABC):

    metrics_id: str

    @abstractmethod
    def eval(self, preds: Sequence[Any], labels: Sequence[Any]) -> Dict[str, float]:
        raise NotImplementedError
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.eval(*args, **kwds)
    
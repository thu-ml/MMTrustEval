from abc import abstractmethod, ABC
from typing import Dict, Any, Sequence, List, Tuple, Union
from mmte.evaluators.metrics import _supported_metrics

class BaseEvaluator(ABC):
    
    evaluator_ids: List[str] = []
    def __init__(self, evaluator_id: str, metrics_cfg: Dict[str, Any]) -> None:
        assert evaluator_id in self.evaluator_ids

        self.evaluator_id = evaluator_id
        self.metrics_cfg = metrics_cfg
        for metrics_id in self.metrics_cfg.keys():
            assert metrics_id in _supported_metrics.keys(), f"{metrics_id} is not supported."

    @abstractmethod
    def process(self, preds: Sequence[Any], labels: Sequence[Any], **kwargs) -> Tuple[Sequence[Any], Sequence[Any]]:
        # no-op
        return preds, labels
    
    def eval(self, preds: Sequence[Any], labels: Sequence[Any], **kwargs) -> Dict[str, Union[Sequence, float]]:
        processed_preds, processed_labels = self.process(preds, labels)
        results = {}

        for metrics_id, kwargs in self.metrics_cfg.items():
            metrics_fn = _supported_metrics[metrics_id]
            keyname = f"{self.__class__.__name__}:{metrics_id}"
            results[keyname] = metrics_fn(processed_labels, processed_preds, **kwargs)
        
        return results
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.eval(*args, **kwds)
    
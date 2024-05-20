from abc import abstractmethod, ABC
from typing import Dict, Any, Sequence, List, Tuple, Union
from mmte.evaluators.metrics import _supported_metrics
from mmte.utils.registry import registry

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
            results[metrics_id] = metrics_fn(processed_labels, processed_preds, **kwargs)
        
        return results
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.eval(*args, **kwds)
    
class SequentialEvaluator:
    
    def __init__(self, evaluator_seq_cfg: Dict[str, Any]) -> None:
        evaluator_seq, evaluator_cls_names = [], []
        for evaluator_id, evaluator_kwargs in evaluator_seq_cfg.items():
            evaluator_cls = registry.get_evaluator_class(evaluator_id)
            evaluator = evaluator_cls(evaluator_id, **evaluator_kwargs)
            evaluator_cls_names.append(evaluator_cls.__name__)
            evaluator_seq.append(evaluator)
        self.evaluator_seq = evaluator_seq
        self.keyname_prefix = "->".join(evaluator_cls_names)
    
    def eval(self, preds: Sequence[Any], labels: Sequence[Any], **kwargs) -> Dict[str, Union[Sequence, float]]:
        for evaluator in self.evaluator_seq[:-1]:
            preds, labels = evaluator.process(preds, labels)   
        
        final_evaluator = self.evaluator_seq[-1]
        results = final_evaluator(preds, labels)
        prefix_results = {f"{self.keyname_prefix}:{key}": value for key, value in results.items()}
        return prefix_results
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.eval(*args, **kwds)
    
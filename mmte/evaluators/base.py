from abc import abstractmethod, ABC
from typing import Dict, Any, Sequence, List, Tuple, Union, Optional
from mmte.evaluators.metrics import _supported_metrics
from mmte.utils.registry import registry

class BaseEvaluator(ABC):
    """
    Base class for evaluators, to evaluate the responses from chatmodels.
    """

    evaluator_ids: List[str] = []
    def __init__(self, evaluator_id: str, metrics_cfg: Dict[str, Any]) -> None:
        """
        Initializing evaluator instance.
        
        Arguments:
            evaluator_id: Identifier for the evaluator
            metrics_cfg: config dict for metrics hooks, format: {metrics_id: metrics_kwargs, ...}
            
        """
        assert evaluator_id in self.evaluator_ids, f"Evaluator {self.evaluator_id} is not available. Only Evaluators in {self.evaluator_ids} can be used."

        self.evaluator_id = evaluator_id
        
        self.metrics_cfg = metrics_cfg
        for metrics_id in self.metrics_cfg.keys():
            assert metrics_id in _supported_metrics.keys(), f"{metrics_id} is not supported."

    @abstractmethod
    def process(self, preds: Sequence[Any], labels: Optional[Sequence[Any]] = None, extras: Optional[Sequence[Any]] = None, **kwargs) -> Tuple[Sequence[Any], Sequence[Any]]:
        """
        1. Perform some processing on sequence data, mainly including scoring/text-extraction with chatmodel/classifier/rule-based, etc.
        2. Different evaluators can be concatenated, and the process function can be cascaded to perform multi-step processing on sequence data.
        
        Arguments:
            preds: responses from chatmodels or preds from `process` function of another evaluator
            labels: groundtruth labels or labels from `process` function of another evaluator
            extras: extra parameters or extra sequence from `process` function of another evaluator
            
        Return:
            preds: processed preds sequence
            labels: processed labels sequence
            extras: processed extra sequence
        """

        # no-op
        return preds, labels, extras
    
    def eval(self, preds: Sequence[Any], labels: Optional[Sequence[Any]] = None, extras: Optional[Sequence[Any]] = None, **kwargs) -> Dict[str, Union[Sequence, float]]:
        """
        Evaluate pipeline including data processing and metrics calculation.
        
        Arguments:
            preds: responses from chatmodels
            labels: groundtruth labels
            extras: extra parameters
            
        Return:
            results
        """

        processed_preds, processed_labels, processed_extras = self.process(preds, labels, extras)
        results = {}

        for metrics_id, kwargs in self.metrics_cfg.items():
            metrics_fn = _supported_metrics[metrics_id]
            results[metrics_id] = metrics_fn(processed_labels, processed_preds, **kwargs)
        
        return results
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.eval(*args, **kwds)
    

class SequentialEvaluator:
    """
    Class for cascading evaluators to perform multi-step processing on sequence data and get results from final sequence data.
    """

    def __init__(self, evaluator_seq_cfg: Dict[str, Any]) -> None:
        """
        Initializing sequence-evaluator instance.
        
        Arguments:
            evaluator_seq_cfg: config dict for instantiatizing evaluators, format: {evaluator: evaluator_kwargs, ...}
            
        """

        evaluator_seq, evaluator_cls_names = [], []
        for evaluator_id, evaluator_kwargs in evaluator_seq_cfg.items():
            evaluator_cls = registry.get_evaluator_class(evaluator_id)
            evaluator = evaluator_cls(evaluator_id, **evaluator_kwargs)
            evaluator_cls_names.append(evaluator_id)
            evaluator_seq.append(evaluator)
        self.evaluator_seq = evaluator_seq
        self.keyname_prefix_seq = self.create_sequence_list(evaluator_cls_names)
    
    def create_sequence_list(self, input_list: List[str]) -> List[str]:
        result = []
        current = ""
        for item in input_list:
            if current:
                current += f"->{item}"
            else:
                current = item
            result.append(current)
        return result

    def eval(self, preds: Sequence[Any], labels: Optional[Sequence[Any]] = None, extras: Optional[Sequence[Any]] = None, **kwargs) -> Dict[str, Union[Sequence, float]]:
        """
        Evaluate pipeline including data processing and metrics calculation.
        
        Arguments:
            preds: responses from chatmodels
            labels: groundtruth labels
            extras: extra parameters
            
        Return:
            results
        """

        prefix_results = {}
        seq_len = len(self.evaluator_seq)
        for evaluator_idx, (evaluator, keyname_prefix) in enumerate(zip(self.evaluator_seq, self.keyname_prefix_seq)):
            if evaluator_idx < seq_len - 1:
                preds, labels, extras = evaluator.process(preds, labels, extras)
                prefix_results.update({f"{keyname_prefix}:pred_no_op": preds})
            else:
                # final evaluator
                results = evaluator(preds, labels, extras)
                prefix_results.update({f"{keyname_prefix}:{key}": value for key, value in results.items()})
        
        return prefix_results
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.eval(*args, **kwds)
    
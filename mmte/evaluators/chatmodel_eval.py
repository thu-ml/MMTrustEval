from typing import Any, Sequence, List, Tuple, Dict
from mmte.evaluators.base import BaseEvaluator
from mmte.utils.registry import registry

@registry.register_evaluator()
class ChatModelEvaluator(BaseEvaluator):
    evaluator_ids: List[str] = ['chatmodel_eval']
    def __init__(self, evaluator_id: str, model_id: str, metrics_cfg: Dict[str, Any], device: str = "cuda") -> None:
        super().__init__(evaluator_id, metrics_cfg)
        from mmte.models import load_chatmodel
        self.chatmodel = load_chatmodel(model_id=model_id, device=device)
    
    def process(self, preds: Sequence[Any], labels: Sequence[Any], **kwargs) -> Tuple[Sequence[Any], Sequence[Any]]:
        processed_preds = []
        for pred in preds:
            pass
        
        return processed_preds, labels
    
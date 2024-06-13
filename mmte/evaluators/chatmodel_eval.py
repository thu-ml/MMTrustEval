from typing import Any, Sequence, List, Tuple, Dict, Optional
from mmte.evaluators.base import BaseEvaluator
from mmte.utils.registry import registry

@registry.register_evaluator()
class ChatModelEvaluator(BaseEvaluator):
    evaluator_ids: List[str] = ['chatmodel_eval_truthfulness', 'chatmodel_eval_fairness']
    def __init__(self, evaluator_id: str, model_id: str, prompt_template: str, generation_kwargs: Dict[str, Any], metrics_cfg: Dict[str, Any], device: str = "cuda") -> None:
        super().__init__(evaluator_id, metrics_cfg)
        from mmte.models import load_chatmodel
        self.chatmodel = load_chatmodel(model_id=model_id, device=device)
        self.prompt_template = prompt_template
        self.generation_kwargs = generation_kwargs
    
    def process(self, preds: Sequence[Any],labels: Optional[Sequence[Any]] = None, **kwargs) -> Tuple[Sequence[Any], Sequence[Any]]:
        processed_preds = []
        for (pred, label) in zip(preds, labels):
            assert isinstance(pred, str)
            if evaluator_id == 'chatmodel_eval_truthfulness':
                eval_prompt = self.prompt_template.format(pred, label)
            elif evaluator_id == 'chatmodel_eval_fairness':
                eval_prompt = self.prompt_template.format(pred)
            eval_message = [
                                {
                                    "role": "user",
                                    "content": eval_prompt
                                }
                            ]
            eval_response = self.chatmodel.chat(messages=eval_message, **self.generation_kwargs)
            print("con",eval_response.content)
            processed_preds.append(eval_response.content)
        return processed_preds, labels
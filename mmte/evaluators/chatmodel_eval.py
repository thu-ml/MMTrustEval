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

@registry.register_evaluator()
class ChatModelScorer(BaseEvaluator):
    evaluator_ids: List[str] = ['chatmodel_score']
    def __init__(self, evaluator_id: str, model_id: str, prompt_template: str, metrics_cfg: Dict[str, Any], full_score: str = "100", device: str = "cuda") -> None:
        super().__init__(evaluator_id, metrics_cfg)
        from mmte.models import load_chatmodel
        self.chatmodel = load_chatmodel(model_id=model_id, device=device)
        self.prompt_template = prompt_template
        self.full_score = full_score
    
    def process(self, preds: Sequence[Any], labels: Sequence[Any], **kwargs) -> Tuple[Sequence[Any], Sequence[Any]]:
        processed_preds = []
        for pred, label in zip(preds, labels):
            label_list = label.split('->')
            if len(label_list)>1:
                question = label_list[0].split('\n')[0]
                label = label_list[1]
                text = self.prompt_template.format(question, label, pred)
            else:
                text = self.prompt_template.format(pred, label)
            
            if pred.strip('\n').lower() == label.lower():    # benchlmm preprocessing
                processed_preds.append(self.full_score)
            else:
                messages = [{
                    "role": "user",
                    "content": text
                    }]
                generation_kwargs = {
                    'max_new_tokens': 20,
                    'do_sample': False,
                }
                response = self.chatmodel.chat(messages, **generation_kwargs)
                response = response.content
                processed_preds.append(response)
        
        return processed_preds, labels

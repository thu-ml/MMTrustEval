from abc import ABC
from typing import Optional, List, Union, Sequence, Any, Dict, Type
from torch.utils.data import DataLoader
from mmte.datasets.base import BaseDataset, collate_fn
from mmte.utils.registry import registry
from mmte.methods.base import BaseMethod
from mmte.models.base import BaseChat
from mmte.evaluators.base import BaseEvaluator
import warnings
import json

class BaseTask(ABC):    
    def __init__(self, task_id: str, dataset_id: str, model_id: str, method_cfg: Optional[Dict] = {}, evaluators_cfg: Dict = {}, log_file: Optional[str] = None) -> None:
        self.dataset_id = dataset_id
        self.model_id = model_id
        self.task_id = task_id
        self.method_cfg = method_cfg
        self.evaluators_cfg = evaluators_cfg
        self.log_file = log_file
    
    def get_handlers(self) -> None:
        self.evaluators = self.get_evaluators()
        self.method = self.get_method() # get method before dataset
        self.dataset = self.get_dataset()
        self.model = self.get_model()
    
    def get_model(self) -> BaseChat:
        model_cls = registry.get_chatmodel_class(self.model_id)
        model = model_cls(self.model_id)
        return model
    
    def get_method(self) -> BaseMethod:
        if not self.method_cfg:
            return None
        
        assert len(self.method_cfg.keys()) == 1
        method_id = list(self.method_cfg.keys())[0]
        method_kwargs = self.method_cfg[method_id]
        method_cls = registry.get_method_class(method_id)
        method = method_cls(method_id, **method_kwargs)
        return method
    
    def get_evaluators(self) -> List[BaseEvaluator]:
        evaluators = []
        for evaluator_id, evaluator_kwargs in self.evaluators_cfg.items():
            evaluator_cls = registry.get_evaluator_class(evaluator_id)
            evaluator = evaluator_cls(evaluator_id, **evaluator_kwargs)
            evaluators.append(evaluator)
        return evaluators

    def get_dataset(self) -> BaseDataset:
        dataset_cls: Type[BaseDataset] = registry.get_dataset_class(self.dataset_id)
        dataset = dataset_cls(self.dataset_id, method_hook=self.method)
        return dataset
    
    def get_dataloader(self) -> DataLoader:
        dataloader = DataLoader(dataset=self.dataset, batch_size=1, collate_fn=collate_fn)
        return dataloader

        
    def eval(self, responses: List[Dict[str, Any]]) -> Dict[str, Union[float, Sequence]]:
        preds: Sequence[str] = [response['response'] for response in responses]
        labels: Sequence[str] = [response['target'] for response in responses]
        results = {}
        for evaluator in self.evaluators:
            result = evaluator(preds, labels)
            for key in result.keys():
                if key in results.keys():
                    warnings.warn(f"{key} already exists in results.")
            results.update(result)
        return results

    def save_results(self, results: Dict[str, Any]) -> None:
        if self.log_file is not None:
            with open(self.log_file, "w") as f:
                json.dump(results, f, indent=4)

    def generate(self, dataloader: DataLoader, **generate_kwargs) -> List[Dict[str, Any]]:
        print('len(self.dataset): ', len(dataloader.dataset))
        responses = []
        for batch_data in dataloader:
            for data in batch_data:
                """
                    # for text data
                    message = [
                        {
                            "role": "user",
                            "content": text
                        }
                    ]

                    # for multimodal data
                    message = [
                        {
                            "role": "user",
                            "content": {
                                "image_path": ...,
                                "text": ...
                            }
                        }
                    ]
                """
                
                message = data['message']
                target = data['target']
                extra: Dict[str, Any] = data['extra']
            
                response = self.model.chat(messages=message, max_new_tokens=50, do_sample=False, **generate_kwargs)
                output = {
                    "content": message[0]['content'],
                    "response": response.content,
                    "target": target,
                    "extra": extra,
                }
            
                responses.append(output)
        
        return responses
        
    def pipeline(self) -> None:
        self.get_handlers()
        dataloader = self.get_dataloader()

        responses = self.generate(dataloader)
        results = self.eval(responses)
        self.save_results(results)
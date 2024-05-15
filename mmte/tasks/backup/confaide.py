from typing import Optional, List, Union
from torch.utils.data import DataLoader
from mmte.datasets.base import BaseDataset
from mmte.metrics.base import BaseDatasetMetrics, BasePerSampleMetrics
from mmte.methods.base import BaseMethod
from mmte.processes.base import BaseProcess
from mmte.models.base import BaseChat
from mmte.tasks.base import BaseTask
from mmte.utils.registry import registry
import json

@registry.register_task()
class ConfAIde_Task(BaseTask):
    
    TASK_DESCRIPTION: str = "confaide-task"
    task_id: str = "confaide-task" # Identifier for the task
    task_ids: List[str] = ["confaide-task"]
    
    supported_model_list: List[str] = ['llava-v1.5-7b', 'minigpt-4-llama2-7b']
    supported_method_list: Optional[List[str]] = None
    supported_dataset_list: List[str] = ["confaide-text", "confaide-image", "confaide-unrelated-image-color", "confaide-unrelated-image-nature", "confaide-unrelated-image-noise"]
    supported_metrics_list: List[str] = ['pearson', 'failure']
    supported_process_list: List[str] = ['txt2score']
    
    def __init__(self, task_id: str, dataset_id: str, model_id: str, method_id: Optional[str], process_id: Optional[str], metrics_id: str) -> None:
        assert dataset_id in self.supported_dataset_list, f"Dataset {dataset_id} is not available. Only datasets in {self.supported_dataset_list} can be used."
        assert metrics_id in self.supported_metrics_list, f"Metrics {metrics_id} is not available. Only metrics in {self.supported_metrics_list} can be used."
        assert model_id in self.supported_model_list, f"Model {model_id} is not available. Only models in {self.supported_model_list} can be used."
        assert method_id is None or \
            method_id in self.supported_method_list, f"Method {self.method_id} is not available. Only methods in {self.supported_method_list} can be used."
        assert process_id is None or \
            process_id in self.supported_process_list, f"Process {self.process_id} is not available. Only processes in {self.supported_process_list} can be used."
        
        self.task_id = task_id
        self.dataset_id = dataset_id
        self.metrics_id = metrics_id
        self.process_id = process_id
        self.method_id = method_id
        self.model_id = model_id
        self.get_handlers()
    
    def get_handlers(self) -> None:
        self.metrics = self.get_metrics()
        self.dataset = self.get_dataset()
        self.method = self.get_method()
        self.process = self.get_process()
        self.model = self.get_model()

    def get_metrics(self) -> List[Union[BaseDatasetMetrics, BasePerSampleMetrics]]:
        metrics_cls = registry.get_metrics_class(self.metrics_id)
        metrics = metrics_cls()
        return metrics
    
    def get_process(self) -> BaseProcess:
        process_cls = registry.get_process_class(self.process_id)
        process = process_cls(self.process_id)
        return process
    
    def get_model(self) -> BaseChat:
        model_cls = registry.get_chatmodel_class(self.model_id)
        model = model_cls(self.model_id)
        return model
    
    def get_method(self) -> BaseMethod:
        if self.method_id is None:
            return None
        method_cls = registry.get_method_class(self.method_id)
        method = method_cls(self.method_id)
        return method

    def get_dataset(self) -> BaseDataset:
        dataset_cls = registry.get_dataset_class(self.dataset_id)
        dataset = dataset_cls(self.dataset_id)
        if self.method_id and self.method:
            poisoned_dataset = self.method(dataset)
            return poisoned_dataset

        return dataset

    def run_task_from_scratch(self, log_file:Optional[str]=None, **kwargs):
        # subclass
        # from torch.utils.data import Subset
        # dataset = Subset(self.dataset, indices=list(range(6)))
        # dataloader = DataLoader(dataset=dataset, batch_size=1, collate_fn=lambda x: x)
        print('len(self.dataset): ', len(self.dataset))
        dataloader = DataLoader(dataset=self.dataset, batch_size=1, collate_fn=lambda x: x)
        responses = []
        for data in dataloader:
            data = data[0]
            text = data.get('text', None)
            image_path = data.get('image_path', None)
            target = data.pop('target', None)

            message = [
                        {
                            "role": "user",
                            "content": data if image_path else text
                        }
                    ]
            response = self.model.chat(messages=message, max_new_tokens=50, do_sample=False)
            raw_output = {"image": image_path, "prompt": text, "response": response.content, "label": target}
            filtered_output = {key: value for key, value in raw_output.items() if value is not None}
            responses.append(filtered_output)

        log = {}
        log['raw_log'] = responses

        preds = [response['response'] for response in responses]
        labels = [response['label'] for response in responses]
        
        if isinstance(self.metrics, BaseDatasetMetrics):
            results = self.metrics(preds, labels, process=self.process)
        elif isinstance(self.metrics, BasePerSampleMetrics):
            results = self.metrics(preds, labels)
        else:
            raise NotImplementedError

        print(results)
        log['result'] = results
        
        if log_file is not None:
            with open(log_file, "w") as f:
                json.dump(log, f, indent=4)
        

    def run_eval_from_json(self, log_file: str, **kwargs):
        
        with open(log_file, "r") as f:
            log = json.load(f)

        responses = log['raw_log']
        preds = [response['response'] for response in responses]
        labels = [response['label'] for response in responses]
        
        if isinstance(self.metrics, BaseDatasetMetrics):
            results = self.metrics(preds, labels, process=self.process)
        elif isinstance(self.metrics, BasePerSampleMetrics):
            results = self.metrics(preds, labels)
        else:
            raise NotImplementedError

        log['result'] = results
        
        if log_file is not None:
            with open(log_file, "w") as f:
                json.dump(log, f, indent=4)

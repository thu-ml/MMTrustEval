from abc import ABC
from typing import Optional, List, Union, Sequence, Any, Dict, Type
from torch.utils.data import DataLoader
from mmte.datasets.base import BaseDataset, collate_fn
from mmte.utils.registry import registry
from mmte.methods.base import BaseMethod
from mmte.models.base import BaseChat
from mmte.evaluators.base import SequentialEvaluator
import numpy as np
import warnings
import json
import os


class BaseTask(ABC):
    def __init__(
        self,
        dataset_id: str,
        model_id: str,
        method_cfg: Optional[Dict] = {},
        dataset_cfg: Optional[Dict] = {},
        generation_kwargs: Optional[Dict] = {},
        evaluator_seq_cfgs: List = [],
        log_file: Optional[str] = None,
    ) -> None:
        self.dataset_id = dataset_id
        self.model_id = model_id
        self.method_cfg = method_cfg
        self.dataset_cfg = dataset_cfg
        self.evaluator_seq_cfgs = evaluator_seq_cfgs
        self.generation_kwargs = generation_kwargs
        self.log_file = log_file

    def get_handlers(self) -> None:
        self.evaluators = self.get_evaluators()
        self.method = self.get_method()  # get method before dataset
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

    def get_evaluators(self) -> List[SequentialEvaluator]:
        evaluators = []

        for evaluator_seq_cfg in self.evaluator_seq_cfgs:
            evaluators.append(SequentialEvaluator(evaluator_seq_cfg=evaluator_seq_cfg))
        return evaluators

    def get_dataset(self) -> BaseDataset:
        dataset_cls: Type[BaseDataset] = registry.get_dataset_class(self.dataset_id)
        dataset = dataset_cls(
            self.dataset_id, method_hook=self.method, **self.dataset_cfg
        )
        return dataset

    def get_dataloader(self) -> DataLoader:
        dataloader = DataLoader(
            dataset=self.dataset, batch_size=1, collate_fn=collate_fn
        )
        return dataloader

    def eval(
        self, responses: List[Dict[str, Any]]
    ) -> Dict[str, Union[float, Sequence]]:
        contents: Sequence[str] = [response["content"] for response in responses]
        preds: Sequence[str] = [response["response"] for response in responses]
        labels: Sequence[str] = [response["target"] for response in responses]
        extras: Sequence[str] = [response["extra"] for response in responses]
        results = {}

        for evaluator in self.evaluators:
            result = evaluator(preds, labels, extras=extras)
            for key in result.keys():
                if key in results.keys():
                    warnings.warn(f"{key} already exists in results.")
            results.update(result)

        # sub aspects
        subset_eval = extras[0] is not None and "subset" in extras[0]
        if subset_eval:
            # Evaluate with subset of dataset, `extra` field in dataclass must have `subset` key to enable subset evaluation.
            subset_list = [item["subset"] for item in extras]
            assert any(subset_list)

            subsets = set(subset_list)
            for subset in subsets:
                preds_subset = [
                    preds[i] for i in range(len(preds)) if subset_list[i] == subset
                ]
                labels_subset = [
                    labels[i] for i in range(len(labels)) if subset_list[i] == subset
                ]
                extras_subset = [
                    extras[i] for i in range(len(extras)) if subset_list[i] == subset
                ]

                subset_results = {}
                for evaluator in self.evaluators:
                    subset_result = evaluator(
                        preds_subset, labels_subset, extras_subset
                    )
                    for key in subset_result.keys():
                        subset_key = f"{key}_{subset}"
                        if subset_key in results.keys():
                            warnings.warn(f"{subset_key} already exists in results.")
                        if not isinstance(subset_result[key], Sequence):
                            # only need summary_keys
                            subset_results[subset_key] = subset_result[key]
                results.update(subset_results)

        results.update(
            {
                "content": contents if any(contents) else None,
                "pred": preds if any(preds) else None,
                "label": labels if any(labels) else None,
                "extra": extras if any(extras) else None,
            }
        )
        return results

    def save_results(self, results: Dict[str, Any]) -> None:
        scatter_keys = [
            key for key, value in results.items() if isinstance(value, Sequence)
        ]
        summary_keys = [
            key
            for key, value in results.items()
            if not isinstance(value, Sequence) and value is not None
        ]

        if scatter_keys:
            seq_len = len(results[scatter_keys[0]])
            for key in scatter_keys:
                assert len(results[key]) == seq_len

        per_sample_results = []
        for idx in range(seq_len):
            per_sample_result = {}
            for key in scatter_keys:
                per_sample_result[key] = results[key][idx]
            per_sample_results.append(per_sample_result)

        formatted_results = {}
        formatted_results["total_results"] = {}
        formatted_results["per_sample_results"] = per_sample_results
        for key in summary_keys:
            formatted_results["total_results"][key] = (
                float(results[key])
                if isinstance(results[key], (np.floating, np.integer))
                else results[key]
            )

        if self.log_file is not None:
            # check if the folder exists
            if not os.path.exists(os.path.dirname(self.log_file)):
                os.makedirs(os.path.dirname(self.log_file))

            with open(self.log_file, "w") as f:
                json.dump(formatted_results, f, indent=4)

    def generate(
        self, dataloader: DataLoader, **generate_kwargs
    ) -> List[Dict[str, Any]]:
        print("len(self.dataset): ", len(dataloader.dataset))
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

                message = data["message"]
                target = data["target"]
                extra: Dict[str, Any] = data["extra"]

                response = self.model.chat(messages=message, **generate_kwargs)
                output = {
                    "content": message[0]["content"],
                    "response": response.content,
                    "target": target,
                    "extra": extra,
                }
                print("output:", output)

                responses.append(output)

        return responses

    def pipeline(self) -> None:
        self.get_handlers()
        dataloader = self.get_dataloader()
        responses = self.generate(dataloader, **self.generation_kwargs)
        results = self.eval(responses)
        self.save_results(results)

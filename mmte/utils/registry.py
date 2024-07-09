class Registry:
    mapping = {
        "chatmodel_name_mapping": {},
        "task_name_mapping": {},
        "paths": {},
        "dataset_name_mapping": {},
        "metrics_name_mapping": {},
        "process_name_mapping": {},
        "method_name_mapping": {},
        "evaluator_name_mapping": {},
    }
    
    
    @classmethod
    def register_chatmodel(cls):
        def wrap(model_cls):
            from mmte.models.base import BaseChat
            assert issubclass(model_cls, BaseChat), "All Chat Models must inherit BaseChat class"
            for model_id in model_cls.model_family:
                if model_id in cls.mapping["chatmodel_name_mapping"]:
                    raise KeyError(
                        "Name '{}' already registered for {}.".format(
                        model_id, cls.mapping["chatmodel_name_mapping"][model_id]
                        )
                    )
                cls.mapping["chatmodel_name_mapping"][model_id] = model_cls
            return model_cls
        
        return wrap
    
    @classmethod
    def get_chatmodel_class(cls, name):
        return cls.mapping["chatmodel_name_mapping"].get(name, None)

    @classmethod
    def list_chatmodels(cls):
        return sorted(cls.mapping["chatmodel_name_mapping"].keys())
        
        
    @classmethod
    def register_path(cls, name, path):
        r"""Register a path to registry with key 'name'

        Args:
            name: Key with which the path will be registered.

        Usage:

            from minigpt4.common.registry import registry
        """
        assert isinstance(path, str), "All path must be str."
        if name in cls.mapping["paths"]:
            raise KeyError("Name '{}' already registered.".format(name))
        cls.mapping["paths"][name] = path
        
    @classmethod
    def get_path(cls, name):
        return cls.mapping["paths"].get(name, None)
    
    
    @classmethod
    def register_task(cls):
        r"""Register a task to registry with key 'name'

        Args:
            name: Key with which the task will be registered.

        Usage:

            from minigpt4.common.registry import registry
        """

        def wrap(task_cls):
            # from mmte.perspectives.base import BaseEval
            from mmte.tasks import BaseTask

            assert issubclass(
                task_cls, BaseTask
            ), "All tasks must inherit BaseTask class"
            for task_id in task_cls.task_ids:
                if task_id in cls.mapping["task_name_mapping"]:
                    raise KeyError(
                        "Name '{}' already registered for {}.".format(
                            task_id, cls.mapping["task_name_mapping"][task_id]
                        )
                    )
                cls.mapping["task_name_mapping"][task_id] = task_cls
            return task_cls

        return wrap
    

    @classmethod
    def register_dataset(cls):
        r"""Register a dataset to registry with key 'name'

        Args:
            name: Key with which the dataset will be registered.
        """

        def wrap(dataset_cls):
            from mmte.datasets.base import BaseDataset

            assert issubclass(
                dataset_cls, BaseDataset
            ), "All tasks must inherit BaseDataset class"
            for dataset_id in dataset_cls.dataset_ids:
                if dataset_id in cls.mapping["dataset_name_mapping"]:
                    raise KeyError(
                        "Name '{}' already registered for {}.".format(
                            dataset_id, cls.mapping["dataset_name_mapping"][dataset_id]
                        )
                    )
                cls.mapping["dataset_name_mapping"][dataset_id] = dataset_cls
            return dataset_cls

        return wrap
    
    @classmethod
    def register_metrics(cls):
        r"""Register a metrics to registry with key 'name'

        Args:
            name: Key with which the metrics will be registered.
        """

        def wrap(metrics_cls):
            from mmte.metrics.base import BaseDatasetMetrics, BasePerSampleMetrics

            assert issubclass(
                metrics_cls, BaseDatasetMetrics
            ) or issubclass(metrics_cls, BasePerSampleMetrics), "All tasks must inherit BaseDatasetMetrics or BasePerSampleMetrics class"
            for metrics_id in metrics_cls.metrics_ids:
                if metrics_id in cls.mapping["metrics_name_mapping"]:
                    raise KeyError(
                        "Name '{}' already registered for {}.".format(
                            metrics_id, cls.mapping["metrics_name_mapping"][metrics_id]
                        )
                    )
                cls.mapping["metrics_name_mapping"][metrics_id] = metrics_cls
            return metrics_cls

        return wrap
    
    @classmethod
    def register_process(cls):
        r"""Register a process to registry with key 'name'

        Args:
            name: Key with which the process will be registered.
        """

        def wrap(process_cls):
            from mmte.processes.base import BaseProcess

            assert issubclass(
                process_cls, BaseProcess
            ), "All tasks must inherit BaseProcess class"
            for process_id in process_cls.process_ids:
                if process_id in cls.mapping["process_name_mapping"]:
                    raise KeyError(
                        "Name '{}' already registered for {}.".format(
                            process_id, cls.mapping["process_name_mapping"][process_id]
                        )
                    )
                cls.mapping["process_name_mapping"][process_id] = process_cls
            return process_cls

        return wrap

    @classmethod
    def register_method(cls):
        r"""Register a method to registry with key 'name'

        Args:
            name: Key with which the method will be registered.
        """

        def wrap(method_cls):
            from mmte.methods.base import BaseMethod

            assert issubclass(
                method_cls, BaseMethod
            ), "All tasks must inherit BaseMethod class"
            for method_id in method_cls.method_ids:
                if method_id in cls.mapping["method_name_mapping"]:
                    raise KeyError(
                        "Name '{}' already registered for {}.".format(
                            method_id, cls.mapping["method_name_mapping"][method_id]
                        )
                    )
                cls.mapping["method_name_mapping"][method_id] = method_cls
            return method_cls

        return wrap
        
    @classmethod
    def register_evaluator(cls):
        r"""Register a evaluator to registry with key 'name'

        Args:
            name: Key with which the evaluator will be registered.
        """

        def wrap(evaluator_cls):
            from mmte.evaluators.base import BaseEvaluator

            assert issubclass(
                evaluator_cls, BaseEvaluator
            ), "All tasks must inherit BaseEvaluator class"
            for evaluator_id in evaluator_cls.evaluator_ids:
                if evaluator_id in cls.mapping["evaluator_name_mapping"]:
                    raise KeyError(
                        "Name '{}' already registered for {}.".format(
                            evaluator_id, cls.mapping["evaluator_name_mapping"][evaluator_id]
                        )
                    )
                cls.mapping["evaluator_name_mapping"][evaluator_id] = evaluator_cls
            return evaluator_cls

        return wrap
    
    @classmethod
    def get_task_class(cls, name):
        return cls.mapping["task_name_mapping"].get(name, None)
    
    @classmethod
    def get_metrics_class(cls, name):
        return cls.mapping["metrics_name_mapping"].get(name, None)
    
    @classmethod
    def get_dataset_class(cls, name):
        return cls.mapping["dataset_name_mapping"].get(name, None)
    
    @classmethod
    def get_process_class(cls, name):
        return cls.mapping["process_name_mapping"].get(name, None)
    
    @classmethod
    def get_method_class(cls, name):
        return cls.mapping["method_name_mapping"].get(name, None)
    
    @classmethod
    def get_evaluator_class(cls, name):
        return cls.mapping["evaluator_name_mapping"].get(name, None)
    
    # @classmethod
    # def list_metrics(cls):
    #     return sorted(cls.mapping["metrics_name_mapping"].keys())
    
    # @classmethod
    # def list_processes(cls):
    #     return sorted(cls.mapping["process_name_mapping"].keys())

    @classmethod
    def list_tasks(cls):
        return sorted(cls.mapping["task_name_mapping"].keys())
    
    @classmethod
    def list_datasets(cls):
        return sorted(cls.mapping["dataset_name_mapping"].keys())
    
    @classmethod
    def list_methods(cls):
        return sorted(cls.mapping["method_name_mapping"].keys())
    
    @classmethod
    def list_evaluators(cls):
        return sorted(cls.mapping["evaluator_name_mapping"].keys())
    
    
        




registry = Registry()
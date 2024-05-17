from mmte.tasks.base import BaseTask
from mmte.utils.registry import registry
from mmte.evaluators.metrics import _supported_metrics
import yaml

print("models: ", registry.list_chatmodels())
print("datasets: ", registry.list_datasets())
print("methods: ", registry.list_methods())
print("evaluators: ", registry.list_evaluators())
print("metrics: ", list(_supported_metrics.keys()))


if __name__ == '__main__':
    configs = [
        "mmte/configs/task/confaide_image.yaml",
        "mmte/configs/task/confaide_text_color.yaml",
        "mmte/configs/task/confaide_text.yaml",
        "mmte/configs/task/vispriv_recognition_vispr.yaml",
        "mmte/configs/task/vispriv_recognition_vizwiz.yaml",
        "mmte/configs/task/visual_leakage_protected.yaml",
        "mmte/configs/task/visual_leakage.yaml",
    ]

    for config in configs:
        with open(config, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
            print(config)
            print(cfg)

            task_id = cfg.get('task_id')
            model_id = cfg.get('model_id')
            dataset_id = cfg.get('dataset_id')
            log_file = cfg.get('log_file')
            method_cfg = cfg.get('method_cfg', {})
            evaluators_cfg = cfg.get('evaluators_cfg', {})
            
            runner = BaseTask(task_id=task_id, dataset_id=dataset_id, model_id=model_id, method_cfg=method_cfg, log_file=log_file, evaluators_cfg=evaluators_cfg)
            runner.pipeline()





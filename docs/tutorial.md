
# Tutorial to add Tasks

以privacy的confaide_text（+unrelated-color-image）为例：

## 定义dataset
定义dataset，并注册dataset_id：confaide-text（可参考`mmte/datasets/confaide.py`）

## 定义evaluator
定义evaluator，并注册evaluator_id：rule_match_and_score_eval
（可参考`mmte/evaluators/rule_eval.py`）

如果需要定义额外的metrics需要自行修改`mmte/evaluators/metrics.py`并在`_supported_metrics`上注册metrics_fn

## 编辑task_config.yaml
（可参考`mmte/configs/task/confaide_text_color.yaml`）

```yaml
dataset_id: "confaide-text"
model_id: "llava-v1.5-7b"

method_cfg:
    {
        unrelated-image-color:
            {
                "img_dir": "./logs/privacy/confaide-text-color",
                "img_size": [50, 50],
                "lazy_mode": True,
            },
    }
log_file: "./logs/privacy/confaide-text-color.json"

evaluator_seq_cfgs:
    [
        {
            "rule_match_and_score_eval":
              { metrics_cfg: { pearson_corr: {}, failure: {} } },
        },
    ]

```

## 一键运行

```shell
python run_task.py --config mmte/configs/task/confaide_text_color.yaml
```


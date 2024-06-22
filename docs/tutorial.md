# Tutorial to Add Tasks

Using the example of Privacy Task P3: infoflow-expectation with dataset `confaide-unrelated-image-color`:

## Define the Dataset
Define the dataset and register the `dataset_id`: `confaide-unrelated-image-color` (refer to `mmte/datasets/confaide.py`).

> If you need to do some preprocessing to your custom dataset, please define `Method` and use it as a hook function in your dataset.

## Define the Evaluator
Define the evaluator and register the `evaluator_id`: `rule_match_and_score_eval` (refer to `mmte/evaluators/rule_eval.py`).

If additional metrics are required, modify `mmte/evaluators/metrics.py` and register the `metrics_fn` in `_supported_metrics`.

## Edit `task_config.yaml`
(refer to `mmte/configs/task/privacy/infoflow.yaml`)

```yaml
dataset_id: "confaide-unrelated-image-color"
model_id: "llava-v1.5-7b"

log_file: "./logs/privacy/infoflow.json"

evaluator_seq_cfgs:
  [
    {
      "rule_match_and_score_eval":
        { metrics_cfg: { pearson_corr: {}, failure: {} } },
    },
  ]


```

## Run with a Single Command

```shell
python run_task.py --config mmte/configs/task/privacy/infoflow.yaml
```

> To modify configurations without changing the yaml file, one can `ADD` or `OVERWRITE` configurations in yaml files using the `--cfg-options` parameter. For example:

```shell
python run_task.py --config mmte/configs/task/privacy/infoflow.yaml --cfg-options dataset_id=confaide-image log_file="logs/privacy/infoflow-confaide-image.json"
```
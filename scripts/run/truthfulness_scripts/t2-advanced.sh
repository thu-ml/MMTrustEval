source activate mllm-dev

dataset_ids=(
    "advanced-spatial"
    "advanced-temporal"
    "advanced-compare"
    "advanced-daily"
    "advanced-traffic"
    "advanced-causality"
    "advanced-math"
    "advanced-code"
    "advanced-translate"
)

for dataset_id in "${dataset_ids[@]}";
do
    python run_task.py --config mmte/configs/task/truthfulness/t2-advanced-inference.yaml --cfg-options dataset_id=${dataset_id} log_file="logs/truthfulness/${dataset_id}.json"
done


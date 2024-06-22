source activate mllm-dev

dataset_ids=(
    "object-detection"
    "risk-analysis"
)

for dataset_id in "${dataset_ids[@]}";
do
    python run_task.py --config mmte/configs/task/safety/risk-identification.yaml --cfg-options dataset_id=${dataset_id} log_file="logs/safety/risk-identification-${dataset_id}.json"
done



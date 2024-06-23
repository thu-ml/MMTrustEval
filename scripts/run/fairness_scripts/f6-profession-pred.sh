source activate mllm-dev

dataset_ids=(
"profession-pred"
"profession-pred-with-description"
)

for dataset_id in "${dataset_ids[@]}";
do
    python run_task.py --config mmte/configs/task/fairness/f6-profession-pred.yaml --cfg-options dataset_id=${dataset_id} log_file="logs/fairness/${dataset_id}.json"
done
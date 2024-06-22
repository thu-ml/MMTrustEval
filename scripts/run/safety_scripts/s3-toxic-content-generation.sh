source activate mllm-dev

dataset_ids=(
    "toxicity-prompt-text"
    "toxicity-prompt-unrelated"
    "toxicity-prompt-image"
)

for dataset_id in "${dataset_ids[@]}";
do
    python run_task.py --config mmte/configs/task/safety/toxic-content-generation.yaml --cfg-options dataset_id=${dataset_id} log_file="logs/safety/toxic-content-generation-${dataset_id}.json"
done






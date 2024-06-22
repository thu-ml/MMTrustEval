source activate mllm-dev

dataset_ids=(
    "typographic-prompt-and-behavior"
    "typographic-prompt"
    "typographic-behavior"
)

for dataset_id in "${dataset_ids[@]}";
do
    python run_task.py --config mmte/configs/task/safety/jailbreaking.yaml --cfg-options dataset_id=${dataset_id} log_file="logs/safety/typographic-jailbreaking-${dataset_id}.json"
done




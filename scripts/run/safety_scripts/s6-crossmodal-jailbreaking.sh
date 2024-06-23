source activate mllm-dev

dataset_ids=(
    "crossmodal-jailbreak-text"
    "crossmodal-jailbreak-unrelated"
    "crossmodal-jailbreak-pos"
    "crossmodal-jailbreak-neg"
)

for dataset_id in "${dataset_ids[@]}";
do
    python run_task.py --config mmte/configs/task/safety/s4-s5-s6-jailbreaking.yaml --cfg-options dataset_id=${dataset_id} log_file="logs/safety/crossmodal-jailbreak-${dataset_id}.json"
done

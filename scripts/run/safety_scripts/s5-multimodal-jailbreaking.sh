source activate mllm-dev

dataset_ids=(
    "opimized-jailbreak-graphic"
    "mm-safety-bench"
    "safebench"
)

for dataset_id in "${dataset_ids[@]}";
do
    python run_task.py --config mmte/configs/task/safety/s4-s5-s6-jailbreaking.yaml --cfg-options dataset_id=${dataset_id} log_file="logs/safety/multimodal-jailbreaking-${dataset_id}.json"
done


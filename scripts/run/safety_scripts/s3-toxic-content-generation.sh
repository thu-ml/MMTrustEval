if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_id>"
    exit 1
fi

model_id=$1

dataset_ids=(
    "toxicity-prompt-text"
    "toxicity-prompt-unrelated"
    "toxicity-prompt-image"
)

for dataset_id in "${dataset_ids[@]}";
do
    python run_task.py --config mmte/configs/task/safety/s3-toxic-content-generation.yaml --cfg-options \
        dataset_id=${dataset_id} \
        model_id=${model_id} \
        log_file="logs/safety/s3-toxic-content-generation/${model_id}/${dataset_id}.json"
done






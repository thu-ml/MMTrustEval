if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_id>"
    exit 1
fi

model_id=$1

dataset_ids=(
    "dt-text"
    "dt-related-image"
    "dt-unrelated-image-color"
    "dt-unrelated-image-nature"
    "dt-unrelated-image-noise"
)

for dataset_id in "${dataset_ids[@]}";
do
    python run_task.py --config mmte/configs/task/robustness/r3-ood-text.yaml --cfg-options \
        dataset_id=${dataset_id} \
        model_id=${model_id} \
        log_file="logs/robustness/r3-ood-text/${model_id}/${dataset_id}.json"
done
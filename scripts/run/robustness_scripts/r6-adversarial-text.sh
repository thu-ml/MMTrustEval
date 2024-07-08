if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_id>"
    exit 1
fi

model_id=$1

dataset_ids=(
    "advglue-text"
    "advglue-related-image"
    "advglue-unrelated-image-color"
    "advglue-unrelated-image-nature"
    "advglue-unrelated-image-noise"
    "advglue-plus-text"
    "advglue-plus-related-image"
    "advglue-plus-unrelated-image-color"
    "advglue-plus-unrelated-image-nature"
    "advglue-plus-unrelated-image-noise"
)

for dataset_id in "${dataset_ids[@]}";
do
    python run_task.py --config mmte/configs/task/robustness/r6-adversarial-text.yaml --cfg-options \
        dataset_id=${dataset_id} \
        model_id=${model_id} \
        log_file="logs/robustness/r6-adversarial-text/${model_id}/${dataset_id}.json"
done

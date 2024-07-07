if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_id>"
    exit 1
fi

model_id=$1

dataset_ids=(
    "g-text-assistance"
    "g-text-misvisual"
    "g-text-unrelated-image-color"
    "g-text-unrelated-image-noise"
    "g-text-unrelated-image-nature"
    "g-text-none"
)

for dataset_id in "${dataset_ids[@]}";
do
    python run_task.py --config mmte/configs/task/truthfulness/t7-visual-misleading.yaml --cfg-options \
        dataset_id=${dataset_id} \
        model_id=${model_id} \
        log_file="logs/truthfulness/t7-visual-misleading/${model_id}/${dataset_id}.json"
done


source activate mllm-dev

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
    python run_task.py --config mmte/configs/task/truthfulness/t7-visual-misleading.yaml --cfg-options dataset_id=${dataset_id} log_file="logs/truthfulness/${dataset_id}.json"
done


source activate mllm-dev

dataset_ids=(
    # "stereo-classification-text"
    # "stereo-classification-image"
    "stereo-classification-unrelated-image-color"
    "stereo-classification-unrelated-image-nature"
    "stereo-classification-unrelated-image-noise"
)

for dataset_id in "${dataset_ids[@]}";
do
    python run_task.py --config mmte/configs/task/fairness/f3-stereo-classification.yaml --cfg-options dataset_id=${dataset_id} log_file="logs/fairness/${dataset_id}.json"
done
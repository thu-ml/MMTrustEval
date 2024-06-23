source activate mllm-dev

dataset_ids=(
    "stereo-agreement-text"
    "stereo-agreement-image"
    "stereo-agreement-unrelated-image-color"
    "stereo-agreement-unrelated-image-nature"
    "stereo-agreement-unrelated-image-noise"
)

for dataset_id in "${dataset_ids[@]}";
do
    python run_task.py --config mmte/configs/task/fairness/f2-stereo-agreement.yaml --cfg-options dataset_id=${dataset_id} log_file="logs/fairness/${dataset_id}.json"
done
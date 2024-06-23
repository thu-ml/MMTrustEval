source activate mllm-dev

dataset_ids=(
    "stereo-query-text"
    "stereo-query-image"
    "stereo-query-unrelated-image-color"
    "stereo-query-unrelated-image-nature"
    "stereo-query-unrelated-image-noise"
)

for dataset_id in "${dataset_ids[@]}";
do
    python run_task.py --config mmte/configs/task/fairness/f4-stereo-query.yaml --cfg-options dataset_id=${dataset_id} log_file="logs/fairness/${dataset_id}.json"
done
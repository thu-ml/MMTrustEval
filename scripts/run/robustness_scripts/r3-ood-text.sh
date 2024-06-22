source activate mllm-dev
dataset_ids=(
    "dt-text"
    "dt-related-image"
    "dt-unrelated-image-color"
    "dt-unrelated-image-nature"
    "dt-unrelated-image-noise"
)

for dataset_id in "${dataset_ids[@]}";
do
    CUDA_VISIBLE_DEVICES=3 python run_task.py --config mmte/configs/task/robustness/ood-text.yaml --cfg-options dataset_id=${dataset_id} log_file="logs/robustness/ood-text-${dataset_id}.json"
done
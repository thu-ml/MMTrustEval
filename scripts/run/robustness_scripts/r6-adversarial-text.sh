source activate mllm-dev
dataset_ids=(
    "advglue-related-image"
    "advglue-unrelated-image-color"
)

for dataset_id in "${dataset_ids[@]}";
do
    CUDA_VISIBLE_DEVICES=1 python run_task.py --config mmte/configs/task/robustness/adversarial-text.yaml --cfg-options dataset_id=${dataset_id} log_file="logs/robustness/adversarial-text-${dataset_id}.json"
done

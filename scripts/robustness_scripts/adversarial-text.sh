source activate mllm-dev
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
    CUDA_VISIBLE_DEVICES=0 python run_task.py --config mmte/configs/task/adversarial-text.yaml --cfg-options dataset_id=${dataset_id} log_file="logs/robustness/adversarial-text-${dataset_id}.json"
done

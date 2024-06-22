source activate mllm-dev
dataset_ids=(
    "confaide-unrelated-image-color"
    "confaide-unrelated-image-nature"
    "confaide-unrelated-image-noise"
    "confaide-text"
    "confaide-image"
)

for dataset_id in "${dataset_ids[@]}";
do
    CUDA_VISIBLE_DEVICES=1 python run_task.py --config mmte/configs/task/privacy/infoflow.yaml --cfg-options dataset_id=${dataset_id} log_file="logs/privacy/infoflow-${dataset_id}.json"
done
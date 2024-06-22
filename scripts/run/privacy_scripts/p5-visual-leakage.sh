source activate mllm-dev
dataset_ids=(
    vispr-leakage
    vispr-leakage-protected
)

for dataset_id in "${dataset_ids[@]}";
do
    CUDA_VISIBLE_DEVICES=1 python run_task.py --config mmte/configs/task/privacy/visual-leakage.yaml --cfg-options dataset_id=${dataset_id} log_file="logs/privacy/visual-leakage-${dataset_id}.json"
done
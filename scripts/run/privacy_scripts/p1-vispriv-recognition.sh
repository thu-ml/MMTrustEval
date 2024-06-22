source activate mllm-dev
dataset_ids=(
    "vispr-recognition"
    "vizwiz-recognition"
)

for dataset_id in "${dataset_ids[@]}";
do
    CUDA_VISIBLE_DEVICES=3 python run_task.py --config mmte/configs/task/privacy/vispriv-recognition.yaml --cfg-options dataset_id=${dataset_id} log_file="logs/privacy/vispriv-recognition-${dataset_id}.json"
done
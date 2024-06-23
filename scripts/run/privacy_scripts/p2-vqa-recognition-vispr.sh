source activate mllm-dev
dataset_ids=(
    "vispr-recognition-vqa-recognition-vispr"
    "vizwiz-drawline-recognition-vqa-recognition-vispr"
    "vizwiz-recognition-vqa-recognition-vispr"
)

for dataset_id in "${dataset_ids[@]}";
do
    CUDA_VISIBLE_DEVICES=3 python run_task.py --config mmte/configs/task/privacy/p2-vqa-recognition-vispr.yaml --cfg-options dataset_id=${dataset_id} log_file="logs/privacy/vqa-recognition-vispr-${dataset_id}.json"
done

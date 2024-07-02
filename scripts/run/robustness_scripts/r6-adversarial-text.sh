source activate mllm-dev

if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_id>"
    exit 1
fi

model_id=$1

dataset_ids=(
    "advglue-related-image"
    "advglue-unrelated-image-color"
)

for dataset_id in "${dataset_ids[@]}";
do
    CUDA_VISIBLE_DEVICES=1 python run_task.py --config mmte/configs/task/robustness/r6-adversarial-text.yaml --cfg-options \
        dataset_id=${dataset_id} \
        model_id=${model_id} \
        log_file="logs/robustness/r6-adversarial-text/${model_id}/${dataset_id}.json"
done

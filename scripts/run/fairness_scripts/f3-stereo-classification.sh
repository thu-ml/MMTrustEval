if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_id>"
    exit 1
fi

model_id=$1

dataset_ids=(
    "stereo-classification-text"
    "stereo-classification-image"
    "stereo-classification-unrelated-image-color"
    "stereo-classification-unrelated-image-nature"
    "stereo-classification-unrelated-image-noise"
)

for dataset_id in "${dataset_ids[@]}";
do
    python run_task.py --config mmte/configs/task/fairness/f3-stereo-classification.yaml --cfg-options \
        dataset_id=${dataset_id} \
        model_id=${model_id} \
        log_file="logs/fairness/f3-stereo-classification/${model_id}/${dataset_id}.json"
done
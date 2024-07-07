if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_id>"
    exit 1
fi

model_id=$1

dataset_ids=(
    "stereo-topic-classification-text"
    "stereo-topic-classification-image"
    "stereo-topic-classification-unrelated-image-color"
    "stereo-topic-classification-unrelated-image-nature"
    "stereo-topic-classification-unrelated-image-noise"
)

for dataset_id in "${dataset_ids[@]}";
do
    python run_task.py --config mmte/configs/task/fairness/f3-stereo-topic-classification.yaml --cfg-options \
        dataset_id=${dataset_id} \
        model_id=${model_id} \
        log_file="logs/fairness/f3-stereo-topic-classification/${model_id}/${dataset_id}.json"
done
if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_id>"
    exit 1
fi

model_id=$1

dataset_ids=(
    "subjective-preference-plain-text"
    "subjective-preference-plain-image"
    "subjective-preference-plain-unrelated-image-color"
    "subjective-preference-plain-unrelated-image-nature"
    "subjective-preference-plain-unrelated-image-noise"
    "subjective-preference-force-text"
    "subjective-preference-force-image"
    "subjective-preference-force-unrelated-image-color"
    "subjective-preference-force-unrelated-image-nature"
    "subjective-preference-force-unrelated-image-noise"
)




for dataset_id in "${dataset_ids[@]}";
do
    python run_task.py --config mmte/configs/task/fairness/f7-subjective-preference.yaml --cfg-options \
        dataset_id=${dataset_id} \
        model_id=${model_id} \
        log_file="logs/fairness/f7-subjective-preference/${model_id}/${dataset_id}.json"
done
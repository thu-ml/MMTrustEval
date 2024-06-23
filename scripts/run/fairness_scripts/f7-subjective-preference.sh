source activate mllm-dev

dataset_ids=(
    "subjective-preference-text"
    "subjective-preference-image"
    "subjective-preference-unrelated-image-color"
    "subjective-preference-unrelated-image-nature"
    "subjective-preference-unrelated-image-noise"
)

for dataset_id in "${dataset_ids[@]}";
do
    python run_task.py --config mmte/configs/task/fairness/f7-subjective-preference.yaml --cfg-options dataset_id=${dataset_id} log_file="logs/fairness/${dataset_id}.json"
done
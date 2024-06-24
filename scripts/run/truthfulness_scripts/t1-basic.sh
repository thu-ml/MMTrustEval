source activate mllm-dev

dataset_ids=(
    # "basic-attribute"
    "basic-grounding"
    # "basic-object"
    # "basic-ocr"
    # "basic-scene"
)

for dataset_id in "${dataset_ids[@]}";
do
    python run_task.py --config mmte/configs/task/truthfulness/t1-${dataset_id}.yaml
done


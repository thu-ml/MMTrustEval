source activate mllm-dev

dataset_ids=(
    "instruction-enhancement-factual"
    "instruction-enhancement-logic"
)

for dataset_id in "${dataset_ids[@]}";
do
    python run_task.py --config mmte/configs/task/truthfulness/t3-${dataset_id}.yaml
done






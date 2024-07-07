if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_id>"
    exit 1
fi

model_id=$1

dataset_ids=(
    "instruction-enhancement-factual"
    "instruction-enhancement-logic"
)

for dataset_id in "${dataset_ids[@]}";
do
    python run_task.py --config mmte/configs/task/truthfulness/t3-${dataset_id}.yaml --cfg-options \
        dataset_id=${dataset_id} \
        model_id=${model_id} \
        log_file="logs/truthfulness/t3-instruction-enhancement/${model_id}/${dataset_id}.json"
done






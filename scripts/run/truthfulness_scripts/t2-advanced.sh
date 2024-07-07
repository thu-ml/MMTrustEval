if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_id>"
    exit 1
fi

model_id=$1

dataset_ids=(
    "advanced-spatial"
    "advanced-temporal"
    "advanced-compare"
    "advanced-daily"
    "advanced-traffic"
    "advanced-causality"
    "advanced-math"
    "advanced-code"
    "advanced-translate"
)

for dataset_id in "${dataset_ids[@]}";
do
    python run_task.py --config mmte/configs/task/truthfulness/t2-advanced-inference.yaml --cfg-options \
        dataset_id=${dataset_id} \
        model_id=${model_id} \
        log_file="logs/truthfulness/t2-advanced/${model_id}/${dataset_id}.json"
done


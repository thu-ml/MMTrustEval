if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_id>"
    exit 1
fi

model_id=$1

dataset_ids=(
    vispr-leakage
    vispr-leakage-protected
)

for dataset_id in "${dataset_ids[@]}";
do
    python run_task.py --config mmte/configs/task/privacy/p5-visual-leakage.yaml --cfg-options \
        dataset_id=${dataset_id} \
        model_id=${model_id} \
        log_file="logs/privacy/p5-visual-leakage/${model_id}/${dataset_id}.json"
done
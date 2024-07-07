if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_id>"
    exit 1
fi

model_id=$1

dataset_id=vision-preference
python run_task.py --config mmte/configs/task/fairness/f5-vision-preference.yaml --cfg-options \
    dataset_id=${dataset_id} \
    model_id=${model_id} \
    log_file="logs/fairness/f5-vision-preference/${model_id}/${dataset_id}.json"

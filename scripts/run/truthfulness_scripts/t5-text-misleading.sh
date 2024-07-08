if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_id>"
    exit 1
fi

model_id=$1

dataset_id=g-text-misleading

python run_task.py --config mmte/configs/task/truthfulness/t5-text-misleading.yaml --cfg-options \
    dataset_id=${dataset_id} \
    model_id=${model_id} \
    log_file="logs/truthfulness/t5-text-misleading/${model_id}/${dataset_id}.json"
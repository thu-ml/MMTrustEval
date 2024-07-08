if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_id>"
    exit 1
fi

model_id=$1

dataset_id=d-mis-visual-confusion

python run_task.py --config mmte/configs/task/truthfulness/t6-visual-confusion.yaml --cfg-options \
    dataset_id=${dataset_id} \
    model_id=${model_id} \
    log_file="logs/truthfulness/t6-visual-confusion/${model_id}/${dataset_id}.json"
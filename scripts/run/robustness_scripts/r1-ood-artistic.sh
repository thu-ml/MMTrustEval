if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_id>"
    exit 1
fi

model_id=$1

dataset_ids=(
    "coco-o-cartoon"
    "coco-o-handmake"
    "coco-o-painting"
    "coco-o-sketch"
    "coco-o-tattoo"
    "coco-o-weather"
)

for dataset_id in "${dataset_ids[@]}";
do
    python run_task.py --config mmte/configs/task/robustness/r1-ood-artistic.yaml --cfg-options \
        dataset_id=${dataset_id} \
        model_id=${model_id} \
        log_file="logs/robustness/r1-ood-artistic/${model_id}/${dataset_id}.json"
done

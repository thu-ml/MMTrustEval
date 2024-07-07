if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_id>"
    exit 1
fi

model_id=$1

dataset_ids=(
    "visual-assistance-text"
    "visual-assistance-image"
    "visual-assistance-unrelated-image-color"
    "visual-assistance-unrelated-image-nature"
    "visual-assistance-unrelated-image-noise"
)

for dataset_id in "${dataset_ids[@]}";
do
    python run_task.py --config mmte/configs/task/truthfulness/t4-visual-assistance.yaml --cfg-options \
        dataset_id=${dataset_id} \
        model_id=${model_id} \
        log_file="logs/truthfulness/t4-visual-assistance/${model_id}/${dataset_id}.json"
done


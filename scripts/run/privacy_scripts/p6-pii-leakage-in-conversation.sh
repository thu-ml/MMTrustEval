if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_id>"
    exit 1
fi

model_id=$1

dataset_ids=(
    "enron-email-text"
    "enron-email-image-oneinfo"
    "enron-email-image-typeinfo"
    "enron-email-image-allinfo"
    "enron-email-unrelated-image-color"
    "enron-email-unrelated-image-nature"
    "enron-email-unrelated-image-noise"
)

for dataset_id in "${dataset_ids[@]}";
do
    python run_task.py --config mmte/configs/task/privacy/p6-pii-leakage-in-conversation.yaml --cfg-options \
        dataset_id=${dataset_id} \
        model_id=${model_id} \
        log_file="logs/privacy/p6-pii-leakage-in-conversation/${model_id}/${dataset_id}.json"
done
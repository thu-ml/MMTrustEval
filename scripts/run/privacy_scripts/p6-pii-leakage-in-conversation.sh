source activate mllm-dev
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
    python run_task.py --config mmte/configs/task/privacy/pii-leakage-in-conversation.yaml --cfg-options dataset_id=${dataset_id} log_file="logs/privacy/pii-leakage-in-conversation-${dataset_id}.json"
done
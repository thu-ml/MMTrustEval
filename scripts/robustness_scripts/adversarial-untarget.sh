source activate mllm-dev
CUDA_VISIBLE_DEVICES=0 python run_task.py --config mmte/configs/task/adversarial-untarget.yaml --cfg-options dataset_id=adv-clean log_file="logs/robustness/adversarial-clean.json"
CUDA_VISIBLE_DEVICES=0 python run_task.py --config mmte/configs/task/adversarial-untarget.yaml --cfg-options dataset_id=adv-untarget log_file="logs/robustness/adversarial-untarget.json"

source activate mllm-dev
CUDA_VISIBLE_DEVICES=3 python run_task.py --config mmte/configs/task/robustness/r4-adversarial-untarget.yaml --cfg-options dataset_id=adv-clean log_file="logs/robustness/adversarial-image-clean.json"
CUDA_VISIBLE_DEVICES=3 python run_task.py --config mmte/configs/task/robustness/r4-adversarial-untarget.yaml --cfg-options dataset_id=adv-untarget log_file="logs/robustness/adversarial-image-untarget.json"

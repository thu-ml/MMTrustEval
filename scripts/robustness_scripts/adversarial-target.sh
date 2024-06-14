source activate mllm-dev
CUDA_VISIBLE_DEVICES=0 python run_task.py --config mmte/configs/task/adversarial-target.yaml --cfg-options dataset_id=adv-target log_file="logs/robustness/adversarial-image-target.json"

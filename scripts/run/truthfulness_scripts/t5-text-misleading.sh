source activate mllm-dev

CUDA_VISIBLE_DEVICES=1 python run_task.py --config mmte/configs/task/truthfulness/t5-text-misleading.yaml
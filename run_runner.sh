

# CUDA_VISIBLE_DEVICES='0' python runner.py \

. env.sh

# export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' 
export CUDA_VISIBLE_DEVICES='0,1,2,3,7' 

python runner.py \
    --task-list all_task.list --task-per-gpu 1 \
    --log-dir run_logs


# python run_task.py --config mmte/configs/task/truthfulness/t1-basic-grounding.yaml --cfg-options         dataset_id=g-basic-grounding         model_id=glm-4v-chat         log_file=logs/truthfulness/t1-basic/glm-4v-chat/g-basic-grounding.json
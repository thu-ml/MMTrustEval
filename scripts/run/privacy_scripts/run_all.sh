source activate mllm-dev

# CUDA_VISIBLE_DEVICES=0
# CUDA_VISIBLE_DEVICES=1
# bash -i scripts/privacy_scripts/confaide.sh &
# wait

# CUDA_VISIBLE_DEVICES=2
CUDA_VISIBLE_DEVICES=0
bash -i scripts/privacy_scripts/pii-leakage-in-context.sh &
# wait

# CUDA_VISIBLE_DEVICES=3
CUDA_VISIBLE_DEVICES=1
bash -i scripts/privacy_scripts/pii-query.sh &
# wait

# CUDA_VISIBLE_DEVICES=4
CUDA_VISIBLE_DEVICES=3
bash -i scripts/privacy_scripts/pri-query.sh &
wait

# CUDA_VISIBLE_DEVICES=5
CUDA_VISIBLE_DEVICES=0
bash -i scripts/privacy_scripts/visual-leakage.sh &
# wait

# CUDA_VISIBLE_DEVICES=6
CUDA_VISIBLE_DEVICES=1
bash -i scripts/privacy_scripts/vispriv-recognition.sh &
wait

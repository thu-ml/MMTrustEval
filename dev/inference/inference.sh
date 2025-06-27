# python test_cambrian.py | tee logs/test_cambrian.log
# python test_internvl2.py | tee logs/test_internvl2.log
# python test_llama3.py | tee logs/test_llama3.log
# python test_molmo.py | tee logs/test_molmo.log
# python test_qwen2.py | tee logs/test_qwen2.log
# python test_llava_hf.py | tee logs/test_llava_hf.log
# python test_llava_next_hf.py | tee logs/test_llava_next_hf.log

. env.sh

# CUDA_VISIBLE_DEVICES=7 python -u inference.py 2>&1 | tee -a all.log
# CUDA_VISIBLE_DEVICES=1 python -u inference.py --use-local
# CUDA_VISIBLE_DEVICES=0 python -u inference.py --model-ids glm-4v-chat
# CUDA_VISIBLE_DEVICES=0 python -u inference.py --model-ids cogvlm2-llama3-chat-19b
# CUDA_VISIBLE_DEVICES=0 python -u inference.py --model-ids deepseek-janus-7b
CUDA_VISIBLE_DEVICES=0 python -u inference.py --model-ids deepseek-vl2

# python test_cambrian.py
(
    export HF_DATASETS_OFFLINE=1 
    export TRANSFORMERS_OFFLINE=1
    # export http_proxy='127.0.0.1:7890'
    # export https_proxy='127.0.0.1:7890'
    # unset http_proxy
    # unset https_proxy

    cd /data/zhangyichi/Trustworthy-MLLM/MMTrustEval
    source activate mllm-dev
        
    model_ids=( 
        minigpt-4-llama2-7b
        minigpt-4-vicuna-13b-v0
        llava-v1.5-7b
        llava-v1.5-13b
        ShareGPT4V-13B
        llava-rlhf-13b
        LVIS-Instruct4V
        otter-mpt-7b-chat
        internlm-xcomposer-7b
        internlm-xcomposer2-vl-7b
        mplug-owl-llama-7b
        mplug-owl2-llama2-7b
        InternVL-Chat-ViT-6B-Vicuna-13B
        qwen-vl-chat
        cogvlm-chat-hf
        llava-v1.6-13b
        instructblip-flan-t5-xxl

        gpt-4-1106-vision-preview
        qwen-vl-plus
        # gemini-pro-vision
        # claude-3-sonnet-20240229
    )
    
    for model_id in "${model_ids[@]}";
    do
        # CUDA_VISIBLE_DEVICES=1 nohup python -u test_run_zw.py --task confaide-info-sensitivity --model_id ${model_id} --output_dir /data/zhangyichi/Trustworthy-MLLM/output/privacy/confaide-force-debug > /data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/confaide_force/confaide-info-sensitivity-on-${model_id}.log 2>&1 &
        # wait
        # CUDA_VISIBLE_DEVICES=1 nohup python -u test_run_zw.py --task confaide-infoflow-expectation-text --model_id ${model_id} --output_dir /data/zhangyichi/Trustworthy-MLLM/output/privacy/confaide-force-debug > /data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/confaide_force/confaide-infoflow-expectation-text-on-${model_id}.log 2>&1 &
        # wait
        CUDA_VISIBLE_DEVICES=1 nohup python -u test_run_zw.py --task confaide-infoflow-expectation-images --model_id ${model_id} --output_dir /data/zhangyichi/Trustworthy-MLLM/output/privacy/confaide-force-debug > /data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/confaide_force/confaide-infoflow-expectation-images-on-${model_id}.log 2>&1 &
        wait
        CUDA_VISIBLE_DEVICES=1 nohup python -u test_run_zw.py --task confaide-infoflow-expectation-unrelated-images-nature --model_id ${model_id} --output_dir /data/zhangyichi/Trustworthy-MLLM/output/privacy/confaide-force-debug > /data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/confaide_force/confaide-infoflow-expectation-unrelated-images-nature-on-${model_id}.log 2>&1 &
        wait
        CUDA_VISIBLE_DEVICES=1 nohup python -u test_run_zw.py --task confaide-infoflow-expectation-unrelated-images-noise --model_id ${model_id} --output_dir /data/zhangyichi/Trustworthy-MLLM/output/privacy/confaide-force-debug > /data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/confaide_force/confaide-infoflow-expectation-unrelated-images-noise-on-${model_id}.log 2>&1 &
        wait
        CUDA_VISIBLE_DEVICES=1 nohup python -u test_run_zw.py --task confaide-infoflow-expectation-unrelated-images-color --model_id ${model_id} --output_dir /data/zhangyichi/Trustworthy-MLLM/output/privacy/confaide-force-debug > /data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/confaide_force/confaide-infoflow-expectation-unrelated-images-color-on-${model_id}.log 2>&1 &
        wait

        # CUDA_VISIBLE_DEVICES=1 python -u -m pdb test_run_zw.py --task confaide-infoflow-expectation-text --model_id ${model_id} --output_dir /data/zhangyichi/Trustworthy-MLLM/output/privacy/confaide-force-debug
    done
)
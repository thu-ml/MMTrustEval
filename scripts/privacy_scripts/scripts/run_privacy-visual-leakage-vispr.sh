(
    # export HF_DATASETS_OFFLINE=1 
    # export TRANSFORMERS_OFFLINE=1

    # unset http_proxy
    # unset https_proxy

    cd /data/zhangyichi/Trustworthy-MLLM/MMTrustEval
    source activate mllm-dev
    
    # model_ids=( 
    #     gpt-4-1106-vision-preview
    #     gemini-pro-vision
    #     claude-3-sonnet-20240229
    #     qwen-vl-plus
    # )

    # for model_id in "${model_ids[@]}";
    # do
    #     CUDA_VISIBLE_DEVICES=6 nohup python -u test_run_zw.py --task visual-leakage-vispr --model_id ${model_id} --output_dir /data/zhangyichi/Trustworthy-MLLM/output/privacy/visual-leakage-vispr-debug >> /data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/visual-leakage-vispr-${model_id}.log 2>&1 &
    # done

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
        gemini-pro-vision
        qwen-vl-plus
        # # claude-3-sonnet-20240229
    )
    
    for model_id in "${model_ids[@]}";
    do
        # CUDA_VISIBLE_DEVICES=0 nohup python -u test_run_zw.py --task visual-leakage-vispr --model_id ${model_id} --output_dir /data/zhangyichi/Trustworthy-MLLM/output/privacy/visual-leakage-vispr-debug > /data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/visual-leakage-vispr-${model_id}.log 2>&1 &
        CUDA_VISIBLE_DEVICES=1 nohup python -u test_run_zw.py --task visual-leakage-vispr-protected --model_id ${model_id} --output_dir /data/zhangyichi/Trustworthy-MLLM/output/privacy/visual-leakage-vispr-protected-debug > /data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/visual-leakage-vispr-protected-${model_id}.log 2>&1 &
        wait
    done

    # model_ids=( 
    #     # "ShareGPT4V-7B"
    #     # "cogvlm-chat-hf"
    #     # "internlm-xcomposer-7b"
    #     # "llava-v1.5-7b"
    #     # "minigpt-4-vicuna-13b-v0"
    #     # "otter-mpt-7b-chat"
    #     # "qwen-vl-chat"
    #     # "mplug-owl2-llama2-7b"
    #     # "minigpt-4-llama2-7b"
    #     "gpt-4-1106-vision-preview"
    # )
    
    # for model_id in "${model_ids[@]}";
    # do
    #     CUDA_VISIBLE_DEVICES=6 nohup python -u test_run_zw.py --task visual-leakage-vispr --model_id ${model_id} --output_dir /data/zhangyichi/Trustworthy-MLLM/output/privacy/visual-leakage-vispr-debug >> /data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/visual-leakage-vispr-${model_id}.log 2>&1 &
    #     wait
    # done
)

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
        # "ShareGPT4V-7B"
        # "cogvlm-chat-hf"
        # "gpt-4-1106-vision-preview"
        # "internlm-xcomposer-7b"
        # "llava-v1.5-7b"
        # "minigpt-4-vicuna-13b-v0"
        # "otter-mpt-7b-chat"
        # "qwen-vl-chat"
        # "mplug-owl2-llama2-7b"
        # "minigpt-4-llama2-7b"


        # gpt-4-1106-vision-preview
        # gemini-pro-vision
        # claude-3-sonnet-20240229
        qwen-vl-plus

        # minigpt-4-llama2-7b
        # minigpt-4-vicuna-13b-v0
        # llava-v1.5-7b
        # llava-v1.5-13b
        # ShareGPT4V-13B
        # llava-rlhf-13b
        # LVIS-Instruct4V
        # otter-mpt-7b-chat
        # internlm-xcomposer-7b
        # internlm-xcomposer2-vl-7b
        # mplug-owl-llama-7b
        # mplug-owl2-llama2-7b
        # InternVL-Chat-ViT-6B-Vicuna-13B
        # qwen-vl-chat
        # cogvlm-chat-hf
        # llava-v1.6-13b
        # instructblip-flan-t5-xxl
    )
    
    task_ids=(
        # confaide-info-sensitivity
        confaide-infoflow-expectation-text
        confaide-infoflow-expectation-images
        confaide-infoflow-expectation-unrelated-images-nature
        confaide-infoflow-expectation-unrelated-images-noise
        confaide-infoflow-expectation-unrelated-images-color
    )

    for model_id in "${model_ids[@]}";
    do
        CUDA_VISIBLE_DEVICES=0 nohup python -u test_run_zw.py --task ${task_ids[0]} --model_id ${model_id} --output_dir "/data/zhangyichi/Trustworthy-MLLM/output/privacy/confaide-debug-sample5" >> "/data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/confaide_sample5/${task_ids[0]}-on-${model_id}.log" 2>&1 &
        wait
        CUDA_VISIBLE_DEVICES=1 nohup python -u test_run_zw.py --task ${task_ids[1]} --model_id ${model_id} --output_dir "/data/zhangyichi/Trustworthy-MLLM/output/privacy/confaide-debug-sample5" >> "/data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/confaide_sample5/${task_ids[1]}-on-${model_id}.log" 2>&1 &
        wait
        CUDA_VISIBLE_DEVICES=5 nohup python -u test_run_zw.py --task ${task_ids[2]} --model_id ${model_id} --output_dir "/data/zhangyichi/Trustworthy-MLLM/output/privacy/confaide-debug-sample5" >> "/data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/confaide_sample5/${task_ids[2]}-on-${model_id}.log" 2>&1 &
        wait
        CUDA_VISIBLE_DEVICES=6 nohup python -u test_run_zw.py --task ${task_ids[3]} --model_id ${model_id} --output_dir "/data/zhangyichi/Trustworthy-MLLM/output/privacy/confaide-debug-sample5" >> "/data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/confaide_sample5/${task_ids[3]}-on-${model_id}.log" 2>&1 &
        wait
        CUDA_VISIBLE_DEVICES=7 nohup python -u test_run_zw.py --task ${task_ids[4]} --model_id ${model_id} --output_dir "/data/zhangyichi/Trustworthy-MLLM/output/privacy/confaide-debug-sample5" >> "/data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/confaide_sample5/${task_ids[4]}-on-${model_id}.log" 2>&1 &
    done

    # model_ids=( 
    #     minigpt-4-llama2-7b
    #     minigpt-4-vicuna-13b-v0
    #     llava-v1.5-7b
    #     llava-v1.5-13b
    #     ShareGPT4V-13B
    #     llava-rlhf-13b
    #     LVIS-Instruct4V
    #     otter-mpt-7b-chat
    #     internlm-xcomposer-7b
    #     internlm-xcomposer2-vl-7b
    #     mplug-owl-llama-7b
    #     mplug-owl2-llama2-7b
    #     InternVL-Chat-ViT-6B-Vicuna-13B
    #     qwen-vl-chat
    #     cogvlm-chat-hf
    #     llava-v1.6-13b
    #     instructblip-flan-t5-xxl
    # )

    # for model_id in "${model_ids[@]}";
    # do
    #     CUDA_VISIBLE_DEVICES=3 nohup python -u test_run_zw.py --task ${task_ids[0]} --model_id ${model_id} --output_dir "/data/zhangyichi/Trustworthy-MLLM/output/privacy/confaide-debug-sample5" >> "/data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/confaide_sample5/${task_ids[0]}-on-${model_id}.log" 2>&1 &
    #     CUDA_VISIBLE_DEVICES=4 nohup python -u test_run_zw.py --task ${task_ids[1]} --model_id ${model_id} --output_dir "/data/zhangyichi/Trustworthy-MLLM/output/privacy/confaide-debug-sample5" >> "/data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/confaide_sample5/${task_ids[1]}-on-${model_id}.log" 2>&1 &
    #     wait
    #     # CUDA_VISIBLE_DEVICES=3 nohup python -u test_run_zw.py --task ${task_ids[2]} --model_id ${model_id} --output_dir "/data/zhangyichi/Trustworthy-MLLM/output/privacy/confaide-debug-sample5" >> "/data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/confaide_sample5/${task_ids[2]}-on-${model_id}.log" 2>&1 &
    #     # CUDA_VISIBLE_DEVICES=4 nohup python -u test_run_zw.py --task ${task_ids[3]} --model_id ${model_id} --output_dir "/data/zhangyichi/Trustworthy-MLLM/output/privacy/confaide-debug-sample5" >> "/data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/confaide_sample5/${task_ids[3]}-on-${model_id}.log" 2>&1 &
    #     # CUDA_VISIBLE_DEVICES=7 nohup python -u test_run_zw.py --task ${task_ids[4]} --model_id ${model_id} --output_dir "/data/zhangyichi/Trustworthy-MLLM/output/privacy/confaide-debug-sample5" >> "/data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/confaide_sample5/${task_ids[4]}-on-${model_id}.log" 2>&1 &
    #     # wait
    # done

    # for model_id in "${model_ids[@]}";
    # do
    #     # CUDA_VISIBLE_DEVICES=3 nohup python -u test_run_zw.py --task ${task_ids[0]} --model_id ${model_id} --output_dir "/data/zhangyichi/Trustworthy-MLLM/output/privacy/confaide-debug-sample5" >> "/data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/confaide_sample5/${task_ids[0]}-on-${model_id}.log" 2>&1 &
    #     # CUDA_VISIBLE_DEVICES=4 nohup python -u test_run_zw.py --task ${task_ids[1]} --model_id ${model_id} --output_dir "/data/zhangyichi/Trustworthy-MLLM/output/privacy/confaide-debug-sample5" >> "/data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/confaide_sample5/${task_ids[1]}-on-${model_id}.log" 2>&1 &
    #     # wait
    #     CUDA_VISIBLE_DEVICES=3 nohup python -u test_run_zw.py --task ${task_ids[2]} --model_id ${model_id} --output_dir "/data/zhangyichi/Trustworthy-MLLM/output/privacy/confaide-debug-sample5" >> "/data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/confaide_sample5/${task_ids[2]}-on-${model_id}.log" 2>&1 &
    #     CUDA_VISIBLE_DEVICES=4 nohup python -u test_run_zw.py --task ${task_ids[3]} --model_id ${model_id} --output_dir "/data/zhangyichi/Trustworthy-MLLM/output/privacy/confaide-debug-sample5" >> "/data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/confaide_sample5/${task_ids[3]}-on-${model_id}.log" 2>&1 &
    #     CUDA_VISIBLE_DEVICES=7 nohup python -u test_run_zw.py --task ${task_ids[4]} --model_id ${model_id} --output_dir "/data/zhangyichi/Trustworthy-MLLM/output/privacy/confaide-debug-sample5" >> "/data/zhangyichi/Trustworthy-MLLM/MMTrustEval/privacy_scripts/logs/confaide_sample5/${task_ids[4]}-on-${model_id}.log" 2>&1 &
    #     wait
    # done

)
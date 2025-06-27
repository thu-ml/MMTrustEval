# readlink -f logs/*/*/*/* | grep phi-3.5 | wc -l
# readlink -f logs/*/*/*/* | grep cambrian-8b | wc -l
# readlink -f logs/*/*/*/* | grep cambrian-13b | wc -l
# readlink -f logs/*/*/*/* | grep molmo-7b-d-0924 | wc -l
# readlink -f logs/*/*/*/* | grep llava-v1.6-mistral-7b-hf | wc -l
# readlink -f logs/*/*/*/* | grep llava-v1.6-vicuna-7b-hf | wc -l
# readlink -f logs/*/*/*/* | grep llava-v1.6-vicuna-13b-hf | wc -l
# readlink -f logs/*/*/*/* | grep llama3-llava-next-8b-hf | wc -l
# readlink -f logs/*/*/*/* | grep llava-1.5-7b-hf | wc -l
# readlink -f logs/*/*/*/* | grep llava-1.5-13b-hf | wc -l
# readlink -f logs/*/*/*/* | grep llama-3-2-chat | wc -l
# readlink -f logs/*/*/*/* | grep internvl2-8b | wc -l
# readlink -f logs/*/*/*/* | grep mplug-owl3-7b-240728 | wc -l
# readlink -f logs/*/*/*/* | grep qwen2-vl-chat | wc -l
# readlink -f logs/*/*/*/* | grep glm-4v-chat | wc -l
# readlink -f logs/*/*/*/* | grep cogvlm2-llama3-chat-19b | wc -l
# readlink -f logs/*/*/*/* | grep deepseek-janus-7b | wc -l
readlink -f logs/*/*/*/* | grep deepseek-vl2 | wc -l



# grep -c -r "Error in generation" logs/ | grep -v ":0$"
# grep -l -r "Error in generation" logs/
# grep -l -r "Error" logs/ | grep -v ":0$"
# grep -c -r "Error" logs/ | grep -v ":0$"
# grep -l -r "Error" logs/
# grep -l -r "Error in generation" logs/ | xargs rm

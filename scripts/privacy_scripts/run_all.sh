
CUDA_VISIBLE_DEVICES=0
bash -i scripts/privacy_scripts/confaide.sh &
CUDA_VISIBLE_DEVICES=2
bash -i scripts/privacy_scripts/pii-leakage-in-context.sh &
CUDA_VISIBLE_DEVICES=3
bash -i scripts/privacy_scripts/pii-query.sh &
CUDA_VISIBLE_DEVICES=4
bash -i scripts/privacy_scripts/pri-query.sh &
CUDA_VISIBLE_DEVICES=5
bash -i scripts/privacy_scripts/visual-leakage.sh &
CUDA_VISIBLE_DEVICES=6
bash -i scripts/privacy_scripts/visual-recongnition.sh &
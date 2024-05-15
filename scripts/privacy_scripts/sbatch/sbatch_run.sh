sbatch -N 1 -p gpu --gres=gpu:1 privacy_scripts/sbatch_privacy_confaide-info-sensitivity.sh &
sbatch -N 1 -p gpu --gres=gpu:1 privacy_scripts/sbatch_privacy_confaide-infoflow-expectation-images.sh &
sbatch -N 1 -p gpu --gres=gpu:1 privacy_scripts/sbatch_privacy_confaide-infoflow-expectation-text.sh &
sbatch -N 1 -p gpu --gres=gpu:1 privacy_scripts/sbatch_privacy_confaide-infoflow-expectation-unrelated-images-color.sh &
sbatch -N 1 -p gpu --gres=gpu:1 privacy_scripts/sbatch_privacy_confaide-infoflow-expectation-unrelated-images-nature.sh &
sbatch -N 1 -p gpu --gres=gpu:1 privacy_scripts/sbatch_privacy_confaide-infoflow-expectation-unrelated-images-noise.sh &
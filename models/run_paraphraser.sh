#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=paraphraser
#SBATCH --gres=gpu:4

python paraphraser.py \
 --model_name_or_path "facebook/bart-large" \
 --dataset_name './style_dataset.py' \
 --dataset_config_name generic \
 --learning_rate 1e-5 \
 --weight_decay 0.1 \
 --num_train_epochs 10 \
 --num_early_stopping 2 \
 --max_seq_length 60 \
 --cache_dir "path_to_your_caches" \
 --output_dir "path_to_your_outdir/paraphraser" \
 --per_device_train_batch_size 6 \
 --overwrite_cache \
 --learning_rate 1e-4 \

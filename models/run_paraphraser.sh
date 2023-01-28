#!/bin/bash
#SBATCH --partition=student,shared,sharedp
#SBATCH --nodes=1
#SBATCH --nodelist=destc0strapp20
#SBATCH --cpus-per-task=10
#SBATCH --job-name=paraphraser
#SBATCH --output=slurm-logs/output.para.nmt.%N.%j.log
#SBATCH --error=slurm-logs/error.para.nmt.%N.%j.log
#SBATCH --gres=gpu:4
#SBATCH --mem=20G
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=ngockhanh.hoang@sony.com

python paraphraser.py \
 --model_name_or_path "facebook/bart-large" \
 --dataset_name './style_dataset.py' \
 --dataset_config_name generic_nmt \
 --learning_rate 1e-5 \
 --weight_decay 0.1 \
 --num_train_epochs 10 \
 --num_early_stopping 2 \
 --max_seq_length 60 \
 --cache_dir "/speech/dbwork/mul/spielwiese3/dehoang/caches" \
 --output_dir "/speech/dbwork/mul/spielwiese3/dehoang/outputs/paraphraser_nmt_lr" \
 --per_device_train_batch_size 6 \
 --overwrite_cache \
 #
 #--model_name_or_path "/speech/dbwork/mul/spielwiese3/dehoang/outputs/paraphraser" \
 #--learning_rate 1e-4 \
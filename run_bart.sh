#!/bin/bash
#SBATCH --partition=student,shared,sharedp
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --nodelist=destc0strapp20
#SBATCH --job-name=bart
#SBATCH --output=slurm-logs/output.bart.emo.nmt.2.0.%N.%j.log
#SBATCH --error=slurm-logs/error.bart.emo.nmt.2.0.%N.%j.log
#SBATCH --gres=gpu:4
#SBATCH --mem=20G
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=ngockhanh.hoang@sony.com

python style_paraphraser.py \
 --model_name_or_path "/speech/dbwork/mul/spielwiese3/dehoang/outputs/paraphraser_nmt09" \
 --dataset_name "./style_dataset.py" \
 --dataset_config_name emo_stylizer \
 --silver_per 2.0 \
 --learning_rate 1e-5 \
 --optimizer "adamw" \
 --num_train_epochs 20 \
 --max_no_improvements 3 \
 --max_length 60 \
 --path_to_classifier_dir "classifier" \
 --cache_dir "/speech/dbwork/mul/spielwiese3/dehoang/caches" \
 --output_path "/speech/dbwork/mul/spielwiese3/dehoang/outputs" \
 --output_dir "style_bart09_nmt_nsl_2.0" \
 --per_device_train_batch_size 6 \
 --per_device_eval_batch_size 1 \
 --gradient_accumulation_steps 2 \
 #--use_style_loss \
 #--use_bertscore \
 # --weight_decay 0.01 \
 #--max_train_steps 500 \
 #--learning_rate 0.001 \
 #
 #neutralizer
 #
 #--model_name_or_path "facebook/bart-large" \

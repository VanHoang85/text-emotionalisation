#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=bart
#SBATCH --gres=gpu:4

python style_paraphraser.py \
 --model_name_or_path "path_to_your_bart_paraphraser" \
 --dataset_name "./style_dataset.py" \
 --dataset_config_name emo_stylizer \
 --silver_per 2.0 \
 --learning_rate 1e-5 \
 --optimizer "adamw" \
 --num_train_epochs 20 \
 --max_no_improvements 3 \
 --max_length 60 \
 --path_to_classifier_dir "path_to_your_classifier" \
 --cache_dir "path_to_your_caches" \
 --output_path "path_to_your_outputs" \
 --output_dir "style_bart09" \
 --per_device_train_batch_size 6 \
 --per_device_eval_batch_size 1 \
 --gradient_accumulation_steps 2 \
 --use_style_loss \
 #--use_bertscore \
 #--learning_rate 0.001 \

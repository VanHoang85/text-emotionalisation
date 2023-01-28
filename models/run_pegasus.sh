#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=pegasus
#SBATCH --gres=gpu:5

python style_paraphraser.py \
 --model_name_or_path "tuner007/pegasus_paraphrase" \
 --dataset_name "./style_dataset.py" \
 --dataset_config_name emo_stylizer \
 --learning_rate 0.0001 \
 --optimizer "adamw" \
 --num_train_epochs 20 \
 --max_no_improvements 3 \
 --max_length 60 \
 --path_to_classifier_dir "path_to_your_classifier" \
 --cache_dir "path_to_your_caches" \
 --output_path "path_to_your_outputs" \
 --output_dir "style_pegasus" \
 --per_device_train_batch_size 6 \
 --per_device_eval_batch_size 1 \
 --gradient_accumulation_steps 2 \
 --use_style_loss \
 #--learning_rate 1e-5 \
 #--model_name_or_path "google/pegasus-large" \

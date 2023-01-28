#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=classifier_train
#SBATCH --gres=gpu:4

python style_classifier.py \
 --model_name_or_path roberta-large \
 --dataset_name './style_dataset.py' \
 --dataset_config_name classifier \
 --do_train \
 --do_eval \
 --learning_rate 2e-5 \
 --num_train_epochs 20 \
 --max_seq_length 128 \
 --cache_dir "path_to_your_caches" \
 --output_dir "./outputs" \
 --per_device_train_batch_size 8 \
 --gradient_accumulation_steps 2 \
 --input_file "../data/style/generic.json" \
 --per_device_eval_batch_size 32 \
 --min_emo_score 0.8 \
 #--evaluation_strategy "steps" \
 #--save_strategy "steps" \
 #--save_steps 1000 \
 #--load_best_model_at_end \
 #--metric_for_best_model "accuracy" \
 #--resume_from_checkpoint "outputs/classifier" \
 #--overwrite_output_dir \
 #--do_predict \
#!/bin/bash
#SBATCH --partition=student,shared,sharedp
#SBATCH --nodes=1
#SBATCH --cpus-per-task=3
#SBATCH --job-name=classifier_predict
#SBATCH --output=slurm-logs/output.class.%N.%j.log
#SBATCH --error=slurm-logs/error.class.%N.%j.log
#SBATCH --gres=gpu:4
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=ngockhanh.hoang@sony.com

python style_classifier.py \
 --model_name_or_path "/speech/dbwork/mul/spielwiese3/dehoang/outputs/classifier" \
 --dataset_name './style_dataset.py' \
 --dataset_config_name classifier \
 --do_predict \
 --learning_rate 2e-5 \
 --num_train_epochs 20 \
 --max_seq_length 128 \
 --cache_dir "/speech/dbwork/mul/spielwiese3/dehoang/caches" \
 --output_dir "./outputs" \
 --per_device_train_batch_size 8 \
 --gradient_accumulation_steps 2 \
 --input_file "./data/style/generic.json" \
 --per_device_eval_batch_size 32 \
 --min_emo_score 0.8 \
 #--overwrite_output_dir \
 #--do_eval \
 #--do_train \
 #--model_name_or_path roberta-large \
 #--output_dir "outputs/classifier" \
 #--evaluation_strategy "steps" \
 #--save_strategy "steps" \
 #--save_steps 1000 \
 #--load_best_model_at_end \
 #--metric_for_best_model "accuracy" \
 #--resume_from_checkpoint "outputs/classifier" \
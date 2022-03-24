#!/bin/bash
#SBATCH --partition=student,shared,sharedp
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --job-name=pegasus
#SBATCH --output=slurm-logs/output.pe.emo.no.sl.%N.%j.log
#SBATCH --error=slurm-logs/error.pe.emo.no.sl.%N.%j.log
#SBATCH --gres=gpu:5
#SBATCH --mem=20G
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=ngockhanh.hoang@sony.com

python style_paraphraser.py \
 --model_name_or_path "tuner007/pegasus_paraphrase" \
 --dataset_name "./style_dataset.py" \
 --dataset_config_name emo_stylizer \
 --learning_rate 0.0001 \
 --optimizer "adamw" \
 --num_train_epochs 20 \
 --max_no_improvements 3 \
 --max_length 60 \
 --path_to_classifier_dir "classifier" \
 --cache_dir "/speech/dbwork/mul/spielwiese3/dehoang/caches" \
 --output_path "/speech/dbwork/mul/spielwiese3/dehoang/outputs" \
 --output_dir "style_pegasus_nsl" \
 --per_device_train_batch_size 6 \
 --per_device_eval_batch_size 1 \
 --gradient_accumulation_steps 2 \
 #--use_style_loss \
 #--max_train_steps 500 \
 #--learning_rate 1e-5 \
 # neutralizer
 #--silver_per 2.0 \
 #--model_name_or_path "/speech/dbwork/mul/spielwiese3/dehoang/outputs/style_pegasus/emo_stylizer" \
 #
 #--model_name_or_path "google/pegasus-large" \

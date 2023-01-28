#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=generation
#SBATCH --gres=gpu:2

python generation.py \
 --model_name_or_path "path_to_your_emo_stylizer" \
 --dataset_name "./style_dataset.py" \
 --dataset_config_name emo_stylizer \
 --path_to_classifier_dir "path_to_your_classifier" \
 --path_to_cola_classifier "path_to_your_cola_classifier" \
 --score_sim \
 --score_grm \
 --score_ppl \
 --num_beams 50 \
 --num_return_sequences 30 \
 --max_pred_return 10 \
 --early_stopping \
 --use_constraint \
 --use_backoff \
 #--gen_all_emos \
 #--input_file "../data/style/new_test.json" \

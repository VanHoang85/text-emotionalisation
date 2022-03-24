#!/bin/bash
#SBATCH --partition=student,shared,sharedp
#SBATCH --nodes=1
#SBATCH --nodelist=destc0strapp10
#SBATCH --cpus-per-task=10
#SBATCH --job-name=generation
#SBATCH --gres=gpu:2
#SBATCH --output=slurm-logs/output.pe.wsl.new.test.%N.%j.log
#SBATCH --error=slurm-logs/error.pe.wsl.new.test.%N.%j.log
#SBATCH --mem=10G
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=ngockhanh.hoang@sony.com

python generation.py \
 --model_name_or_path "/speech/dbwork/mul/spielwiese3/dehoang/outputs/style_pegasus/emo_stylizer" \
 --dataset_name "./style_dataset.py" \
 --dataset_config_name emo_stylizer \
 --input_file "data/style/new_test.json" \
 --output_file "pegasus_wsl_new_test.json" \
 --score_sim \
 --score_grm \
 --score_ppl \
 --num_beams 50 \
 --num_return_sequences 30 \
 --max_pred_return 10 \
 --early_stopping \
 --use_constraint \
 --use_backoff \
 --gen_all_emos \
 #
 #
 #--dataset_config_name neutralizer \
 #--output_file "naive_phrases_test_emo_wbf.json" \
 #
 #--output_file "pegasus_test_all_emos.json" \
 #--output_file "pegasus_with_style_loss_test_emo_nms.json" \
 #--model_name_or_path "naive" \
 #
 #--model_name_or_path "target" \
 #--model_name_or_path "/speech/dbwork/mul/spielwiese3/dehoang/outputs/style_bart09_nsl/emo_stylizer" \
 #--model_name_or_path "/speech/dbwork/mul/spielwiese3/dehoang/outputs/style_bart09/emo_stylizer" \
 #--model_name_or_path "/speech/dbwork/mul/spielwiese3/dehoang/outputs/style_bart09/neutralizer" \
 #--model_name_or_path "/speech/dbwork/mul/spielwiese3/dehoang/outputs/style_bart09_no_para/emo_stylizer" \
 #--model_name_or_path "tuner007/pegasus_paraphrase" \
 #--model_name_or_path "/speech/dbwork/mul/spielwiese3/dehoang/outputs/style_pegasus/neutralizer" \
 #--model_name_or_path "/speech/dbwork/mul/spielwiese3/dehoang/outputs/style_pegasus_nsl/emo_stylizer" \
 #--model_name_or_path "/speech/dbwork/mul/spielwiese3/dehoang/outputs/style_pegasus_no_para/emo_stylizer" \
 #--output_file "pegasus_new_test_all_emos.json" \
 #--model_name_or_path "/speech/dbwork/mul/spielwiese3/dehoang/outputs/style_bart09_nmt_nsl/emo_stylizer2.0" \
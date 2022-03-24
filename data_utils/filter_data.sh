#!/bin/bash
#SBATCH --partition=student,shared,sharedp
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --job-name=filter
#SBATCH --gres=gpu:4
#SBATCH --mem=10G
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=ngockhanh.hoang@sony.com

python filter_data.py
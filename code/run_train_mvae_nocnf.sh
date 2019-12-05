#!/usr/bin/env bash
#$ -l cuda=1   # request one GPU (remove this line if none is needed)
#$ -q all.q    # don't fill the qlogin queue

cd /home/MihailBogojeski/git/multimodaldiffeq/code
. /home/MihailBogojeski/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda activate mult_diffeq

# python train_ts_mvae.py -e 500 --batch_size 128 --z_size 64 --hidden_size 128 --annealing-epochs 50 -lr 1e-3
python train_ts_mvae.py -e 500 --batch_size 256 --z_size 64 --hidden_size 128 --annealing-epochs 50 -lr 1e-3

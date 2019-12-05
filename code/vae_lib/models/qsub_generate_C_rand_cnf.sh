#!/usr/bin/env bash
#$ -l cuda=1   # request one GPU (remove this line if none is needed)
#$ -q all.q    # don't fill the qlogin queue

cd /home/MihailBogojeski/git/multimodaldiffeq/code/vae_lib/models
. /home/MihailBogojeski/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda activate mult_diffeq

python ts_sample.py -d plant_c --n_samples 1500 --model_path ../../trained_ts_models/model_best_c_cnf.pth.tar --length 120 --length_std 10 --flow
mv ts_sample.npy ts_sample_rand_c_cnf_nocon.npy
python ts_sample.py -d plant_c --n_samples 1500 --model_path ../../trained_ts_models/model_best_c_cnf.pth.tar --length 120 --condition-on-targets 1 --condition-on-conditions 1 --flow
mv ts_sample.npy ts_sample_rand_c_cnf_allcon.npy
mv ts_target.npy ts_rand_c_cnf_alltarget.npy
mv ts_cond.npy ts_rand_c_cnf_allcond.npy
python ts_sample.py -d plant_c --n_samples 1500 --model_path ../../trained_ts_models/model_best_c_cnf.pth.tar --length 120 --condition-on-conditions 1 --flow
mv ts_sample.npy ts_sample_rand_c_cnf_concon.npy
mv ts_cond.npy ts_rand_c_cnf_cond.npy
python ts_sample.py -d plant_c --n_samples 1500 --model_path ../../trained_ts_models/model_best_c_cnf.pth.tar --length 120 --condition-on-targets 1 --flow
mv ts_sample.npy ts_sample_rand_c_cnf_tarcon.npy
mv ts_target.npy ts_rand_c_cnf_target.npy

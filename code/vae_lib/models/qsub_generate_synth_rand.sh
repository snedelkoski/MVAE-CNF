#!/usr/bin/env bash
#$ -l cuda=1   # request one GPU (remove this line if none is needed)
#$ -q all.q    # don't fill the qlogin queue

cd /home/MihailBogojeski/git/multimodaldiffeq/code/vae_lib/models
. /home/MihailBogojeski/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda activate mult_diffeq


python ts_sample.py -d synthetic --n_samples 32 --model_path ../../trained_ts_models/model_best_s2_nocnf.pth.tar --length 200
mv ts_sample.npy ts_sample_rand_s2_nocnf_nocon.npy
python ts_sample.py -d synthetic --n_samples 32 --model_path ../../trained_ts_models/model_best_s2_nocnf.pth.tar --length 200 --condition-on-targets 1 --condition-on-conditions 1
mv ts_sample.npy ts_sample_rand_s2_nocnf_allcon.npy
mv ts_cond.npy ts_rand_s2_nocnf_allcond.npy
mv ts_target.npy ts_rand_s2_nocnf_alltarget.npy
python ts_sample.py -d synthetic --n_samples 32 --model_path ../../trained_ts_models/model_best_s2_nocnf.pth.tar --length 200 --condition-on-conditions 1
mv ts_sample.npy ts_sample_rand_s2_nocnf_concon.npy
mv ts_cond.npy ts_rand_s2_nocnf_cond.npy
python ts_sample.py -d synthetic --n_samples 32 --model_path ../../trained_ts_models/model_best_s2_nocnf.pth.tar --length 200 --condition-on-targets 1
mv ts_sample.npy ts_sample_rand_s2_nocnf_tarcon.npy
mv ts_target.npy ts_rand_s2_nocnf_target.npy

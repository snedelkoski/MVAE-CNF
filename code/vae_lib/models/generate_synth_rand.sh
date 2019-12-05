# python ts_sample.py -d synthetic --n_samples 32 --model_path ../../trained_ts_models/model_best_s.pth.tar --length 200
# mv ts_sample.npy ts_sample_rand_s_nocon.npy
# python ts_sample.py -d synthetic --n_samples 32 --model_path ../../trained_ts_models/model_best_s.pth.tar --length 200 --condition-on-targets 1 --condition-on-conditions 1
# mv ts_sample.npy ts_sample_rand_s_allcon.npy
# mv ts_cond.npy ts_rand_s_allcond.npy
# mv ts_target.npy ts_rand_s_alltarget.npy
# python ts_sample.py -d synthetic --n_samples 32 --model_path ../../trained_ts_models/model_best_s.pth.tar --length 200 --condition-on-conditions 1
# mv ts_sample.npy ts_sample_rand_s_concon.npy
# mv ts_cond.npy ts_rand_s_cond.npy
# python ts_sample.py -d synthetic --n_samples 32 --model_path ../../trained_ts_models/model_best_s.pth.tar --length 200 --condition-on-targets 1
# mv ts_sample.npy ts_sample_rand_s_tarcon.npy
# mv ts_target.npy ts_rand_s_target.npy


python ts_sample.py -d synthetic --n_samples 32 --model_path ../../trained_ts_models/model_best_s_nocnf.pth.tar --length 200
mv ts_sample.npy ts_sample_rand_s_nocnf_nocon.npy
python ts_sample.py -d synthetic --n_samples 32 --model_path ../../trained_ts_models/model_best_s_nocnf.pth.tar --length 200 --condition-on-targets 1 --condition-on-conditions 1
mv ts_sample.npy ts_sample_rand_s_nocnf_allcon.npy
mv ts_cond.npy ts_rand_s_nocnf_allcond.npy
mv ts_target.npy ts_rand_s_nocnf_alltarget.npy
python ts_sample.py -d synthetic --n_samples 32 --model_path ../../trained_ts_models/model_best_s_nocnf.pth.tar --length 200 --condition-on-conditions 1
mv ts_sample.npy ts_sample_rand_s_nocnf_concon.npy
mv ts_cond.npy ts_rand_s_nocnf_cond.npy
python ts_sample.py -d synthetic --n_samples 32 --model_path ../../trained_ts_models/model_best_s_nocnf.pth.tar --length 200 --condition-on-targets 1
mv ts_sample.npy ts_sample_rand_s_nocnf_tarcon.npy
mv ts_target.npy ts_rand_s_nocnf_target.npy

python ts_sample.py -d synthetic --n_samples 32 --model_path ../../trained_ts_models/model_best_s_nocnf.pth.tar --length 200 --smooth True
mv ts_sample.npy ts_sample_smooth_s_nocnf_nocon.npy
python ts_sample.py -d synthetic --n_samples 32 --model_path ../../trained_ts_models/model_best_s_nocnf.pth.tar --length 200 --condition-on-targets 1 --condition-on-conditions 1 --smooth True
mv ts_sample.npy ts_sample_smooth_s_nocnf_allcon.npy
mv ts_cond.npy ts_smooth_s_nocnf_allcond.npy
mv ts_target.npy ts_smooth_s_nocnf_alltarget.npy
python ts_sample.py -d synthetic --n_samples 32 --model_path ../../trained_ts_models/model_best_s_nocnf.pth.tar --length 200 --condition-on-conditions 1 --smooth True
mv ts_sample.npy ts_sample_smooth_s_nocnf_concon.npy
mv ts_cond.npy ts_smooth_s_nocnf_cond.npy
python ts_sample.py -d synthetic --n_samples 32 --model_path ../../trained_ts_models/model_best_s_nocnf.pth.tar --length 200 --condition-on-targets 1 --smooth True
mv ts_sample.npy ts_sample_smooth_s_nocnf_tarcon.npy
mv ts_target.npy ts_smooth_s_nocnf_target.npy


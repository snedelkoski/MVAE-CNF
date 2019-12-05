python ts_simple_sample.py -d synthetic --n_samples 32 --model_path ../../trained_simple_ts_models/model_best.pth.tar --length 200
mv ts_sample.npy ts_simple_s_nocon.npy
python ts_simple_sample.py -d synthetic --n_samples 32 --model_path ../../trained_simple_ts_models/model_best.pth.tar --length 200 --condition-on-targets 1 --condition-on-conditions 1
mv ts_sample.npy ts_simple_s_allcon.npy
mv ts_cond.npy ts_r_simple_allcond.npy
mv ts_target.npy ts_r_simple_alltarget.npy
python ts_simple_sample.py -d synthetic --n_samples 32 --model_path ../../trained_simple_ts_models/model_best.pth.tar --length 200 --condition-on-conditions 1
mv ts_sample.npy ts_simple_s_concon.npy
mv ts_cond.npy ts_r_simple_cond.npy
python ts_simple_sample.py -d synthetic --n_samples 32 --model_path ../../trained_simple_ts_models/model_best.pth.tar --length 200 --condition-on-targets 1
mv ts_sample.npy ts_simple_s_tarcon.npy
mv ts_target.npy ts_r_simple_target.npy


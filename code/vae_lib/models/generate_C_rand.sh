python ts_sample.py -d plant_c --n_samples 1500 --model_path ../../trained_ts_models/model_best_c_nocnf.pth.tar --length 120 --length_std 10
mv ts_sample.npy ts_sample_rand_c_nocnf_nocon.npy
python ts_sample.py -d plant_c --n_samples 1500 --model_path ../../trained_ts_models/model_best_c_nocnf.pth.tar --length 120 --condition-on-targets 1 --condition-on-conditions 1
mv ts_sample.npy ts_sample_rand_c_nocnf_allcon.npy
mv ts_target.npy ts_rand_c_nocnf_alltarget.npy
mv ts_cond.npy ts_rand_c_nocnf_allcond.npy
python ts_sample.py -d plant_c --n_samples 1500 --model_path ../../trained_ts_models/model_best_c_nocnf.pth.tar --length 120 --condition-on-conditions 1
mv ts_sample.npy ts_sample_rand_c_nocnf_concon.npy
mv ts_cond.npy ts_rand_c_nocnf_cond.npy
python ts_sample.py -d plant_c --n_samples 1500 --model_path ../../trained_ts_models/model_best_c_nocnf.pth.tar --length 120 --condition-on-targets 1
mv ts_sample.npy ts_sample_rand_c_nocnf_tarcon.npy
mv ts_target.npy ts_rand_c_nocnf_target.npy

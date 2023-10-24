rm -r data/MultiWOZ_2.0/cross-ws100-processed

python3 devide_by_domain.py -target_domains taxi-hospital-attraction

python3 preprocess.py -version 2.0 -sum_window 100

python3 main.py -version 2.0 -run_type train -backbone model_path/ -model_dir ./v0_single -grad_accum_steps 2 -batch_size 4 -ururu -warmup_ratio 0.1 -woz_type cross -no_validation
python3 main.py -run_type predict -ckpt v0_single/ckpt-epoch1 -output predict.json -batch_size 64
python3 main.py -run_type predict -ckpt v0_single/ckpt-epoch2 -output predict.json -batch_size 64
python3 main.py -run_type predict -ckpt v0_single/ckpt-epoch3 -output predict.json -batch_size 64
python3 main.py -run_type predict -ckpt v0_single/ckpt-epoch4 -output predict.json -batch_size 64
python3 main.py -run_type predict -ckpt v0_single/ckpt-epoch5 -output predict.json -batch_size 64
python3 main.py -run_type predict -ckpt v0_single/ckpt-epoch6 -output predict.json -batch_size 64
python3 main.py -run_type predict -ckpt v0_single/ckpt-epoch7 -output predict.json -batch_size 64
python3 main.py -run_type predict -ckpt v0_single/ckpt-epoch8 -output predict.json -batch_size 64
python3 main.py -run_type predict -ckpt v0_single/ckpt-epoch9 -output predict.json -batch_size 64
python3 main.py -run_type predict -ckpt v0_single/ckpt-epoch10 -output predict.json -batch_size 64
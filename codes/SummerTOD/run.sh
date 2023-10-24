python preprocess.py

python3 main.py -run_type train -backbone model_path/ -version 2.1 -model_dir ./v1-resp+bs-cs3-ss100 -grad_accum_steps 2 -batch_size 4 -ururu -warmup_ratio 0.1
python3 main.py -run_type predict -ckpt v1-resp+bs-cs3-ss100/ckpt-epoch1 -output predict.json -batch_size 128
python3 main.py -run_type predict -ckpt v1-resp+bs-cs3-ss100/ckpt-epoch2 -output predict.json -batch_size 128
python3 main.py -run_type predict -ckpt v1-resp+bs-cs3-ss100/ckpt-epoch3 -output predict.json -batch_size 128
python3 main.py -run_type predict -ckpt v1-resp+bs-cs3-ss100/ckpt-epoch4 -output predict.json -batch_size 128
python3 main.py -run_type predict -ckpt v1-resp+bs-cs3-ss100/ckpt-epoch5 -output predict.json -batch_size 128
python3 main.py -run_type predict -ckpt v1-resp+bs-cs3-ss100/ckpt-epoch6 -output predict.json -batch_size 128
python3 main.py -run_type predict -ckpt v1-resp+bs-cs3-ss100/ckpt-epoch7 -output predict.json -batch_size 128
python3 main.py -run_type predict -ckpt v1-resp+bs-cs3-ss100/ckpt-epoch8 -output predict.json -batch_size 128
python3 main.py -run_type predict -ckpt v1-resp+bs-cs3-ss100/ckpt-epoch9 -output predict.json -batch_size 128
python3 main.py -run_type predict -ckpt v1-resp+bs-cs3-ss100/ckpt-epoch10 -output predict.json -batch_size 128
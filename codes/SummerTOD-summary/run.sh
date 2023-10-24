python3 main.py -run_type train -version 2.0 -backbone model_path/ -model_dir ./v0-sum-ss100_layer1_nohistory -summary_context_size 101 -ururu
python3 main.py -run_type predict -ckpt v0-sum-ss100_layer1_nohistory/ckpt-epoch1 -output predict.json -batch_size 128
python3 main.py -run_type predict -ckpt v0-sum-ss100_layer1_nohistory/ckpt-epoch2 -output predict.json -batch_size 128
python3 main.py -run_type predict -ckpt v0-sum-ss100_layer1_nohistory/ckpt-epoch3 -output predict.json -batch_size 128
python3 main.py -run_type predict -ckpt v0-sum-ss100_layer1_nohistory/ckpt-epoch4 -output predict.json -batch_size 128
python3 main.py -run_type predict -ckpt v0-sum-ss100_layer1_nohistory/ckpt-epoch5 -output predict.json -batch_size 128
python3 main.py -run_type predict -ckpt v0-sum-ss100_layer1_nohistory/ckpt-epoch6 -output predict.json -batch_size 128
python3 main.py -run_type predict -ckpt v0-sum-ss100_layer1_nohistory/ckpt-epoch7 -output predict.json -batch_size 128
python3 main.py -run_type predict -ckpt v0-sum-ss100_layer1_nohistory/ckpt-epoch8 -output predict.json -batch_size 128
python3 main.py -run_type predict -ckpt v0-sum-ss100_layer1_nohistory/ckpt-epoch9 -output predict.json -batch_size 128
python3 main.py -run_type predict -ckpt v0-sum-ss100_layer1_nohistory/ckpt-epoch10 -output predict.json -batch_size 128
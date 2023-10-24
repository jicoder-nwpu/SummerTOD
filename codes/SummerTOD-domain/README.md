## Environment setting

Our python version is 3.7.5.

The package can be installed by running the following command.

```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Data Preprocessing

For the experiments, we use MultiWOZ2.0 and MultiWOZ2.1.
- (MultiWOZ2.0) annotated_user_da_with_span_full.json: A fully annotated version of the original MultiWOZ2.0 data released by developers of Convlab available [here](https://github.com/ConvLab/ConvLab/tree/master/data/multiwoz/annotation).
- (MultiWOZ2.1) data.json: The original MultiWOZ 2.1 data released by researchers in University of Cambrige available [here](https://github.com/budzianowski/multiwoz/tree/master/data).
- (MultiWOZ2.2) data.json: The MultiWOZ2.2 dataset converted to the same format as MultiWOZ2.1 using script [here](https://github.com/budzianowski/multiwoz/tree/master/data/MultiWOZ_2.2).

We use the preprocessing scripts implemented by [Zhang et al., 2020](https://arxiv.org/abs/1911.10484). Please refer to [here](https://github.com/thu-spmi/damd-multiwoz/blob/master/data/multi-woz/README.md) for the details.

```
python preprocess.py -version $VERSION -sum_window $WINDOWS_SIZE
```

## Training

Our implementation supports a single GPU. Please use smaller batch sizes if out-of-memory error raises.

- SummerTOD
```
python3 main.py -version 2.0 -run_type train -backbone model_path/ -model_dir ./v0_single -grad_accum_steps 2 -batch_size 4 -ururu -warmup_ratio 0.1 -woz_type cross -no_validation
```

- The checkpoints will be saved at the end of each epoch (the default training epoch is set to 10).
## Inference

```
python main.py -run_type predict -ckpt $CHECKPOINT -output $MODEL_OUTPUT -batch_size $BATCH_SIZE
```

All checkpoints are saved in ```$MODEL_DIR``` with names such as 'ckpt-epoch10'.

The result file (```$MODEL_OUTPUT```) will be saved in the checkpoint directory.

To reduce inference time, it is recommended to set large ```$BATCH_SIZE```. In our experiemnts, it is set to 16 for inference.

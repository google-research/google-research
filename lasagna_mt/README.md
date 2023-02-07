# Lasagna: Stacking Diverse Architectures to Improve Machine Translation

Authors: Andrea Schioppa, Nal Kalchbrenner.

To Appear in TMLR-2023.

Note, we currently just release the code for Auto-Regressive models for the
fp32 training use case.

## Installation
The model code is meant to work with [fairseq](https://github.com/facebookresearch/fairseq).
There are two options to run the models:

1. Install fairseq. Then copy `lasagna.py` to 
`fairseq/fairseq/models/lasagna.py`. 
2. Create your own training script. Make sure to import `lasagna.py` at the
top of it. You might need to adjust the import paths in the file. An
example of this approach is illustrated by the fairseq example in
[torchscale](https://github.com/microsoft/torchscale).

In any case, when installing fairseq you will need to compile the CUDA kernels
for Light and Dynamic convolutions, e.g.

```bash
fairseq/fairseq/modules/{lightconv_layer, dynamicconv_layer}
CUDA_HOME=/usr/local/cuda python3 cuda_function_gen.py
CUDA_HOME=/usr/local/cuda python3 setup.py install
```

## Using the model
The implementation of `lasagna.py` allows to decide which kind of layer
to use at a given position in the Encoder or the Decoder. For example,
for the Lasagna model described in the paper:

```bash
python3 -m torch.distributed.launch --nproc_per_node 1 $(which fairseq-train) \
  ${DATA_DIR} \
  --share-all-embeddings --max-update 120000   \
  --adam-betas '(0.9, 0.98)' --keep-last-epochs 2 \
  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
  --warmup-init-lr 1e-07 \
  --clip-norm 0 --optimizer adam  \
  --stop-min-lr 1e-09 --update-freq 1 \
  --source-lang en --target-lang de --max-tokens 4000 --no-progress-bar \
  --log-interval 100 --weight-decay 0.0 \
  --max-source-positions 256 \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --ddp-backend=legacy_ddp \
  -a lasagna_wmt_en_de  \
  --save-dir checkpoints --num-workers 2 \
  --tensorboard-logdir tb-logs \
  --dropout 0.3 --attention-dropout 0.1  --weight-dropout 0.1 \
  --best-checkpoint-metric loss \
  --encoder-layer-type-list '["conv", "conv", "gmlp", "gmlp", "self", "self", "self"]' \
  --encoder-conv-use-glu-list '[0, 0, 0, 0, 0, 0, 0]' \
  --encoder-kernel-size-list '[31, 31, 31, 31, 31, 31, 31]' \
  --decoder-layer-type-list '["gmlp", "gmlp", "gmlp", "gmlp", "gmlp", "gmlp"]' \
  --decoder-kernel-size-list '[31, 31, 31, 31, 31, 31]' \
  --decoder-conv-use-glu-list '[0, 0, 0, 0, 0, 0]'
```

For example, if one wants to switch to Dynamic convolutions:

1. Modify the `-layer-type-list` using "conv";
2. Enable the GLU in all the layers.
3. Set the convolution type to dynamic.

```bash
python3 -m torch.distributed.launch --nproc_per_node 1 $(which fairseq-train) \
  ${DATA_DIR} \
  --share-all-embeddings --max-update 120000   \
  --adam-betas '(0.9, 0.98)' --keep-last-epochs 2 \
  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
  --clip-norm 0 --optimizer adam  \
  --stop-min-lr 1e-09 --update-freq 1 \
  --source-lang en --target-lang de --max-tokens 4000 --no-progress-bar \
  --log-interval 100 --weight-decay 0.0 \
  --max-source-positions 256 \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --ddp-backend=legacy_ddp \
  -a lasagna_wmt_en_de  \
  --save-dir checkpoints --num-workers 2 \
  --tensorboard-logdir tb-logs \
  --dropout 0.3 --attention-dropout 0.1  --weight-dropout 0.1 \
  --best-checkpoint-metric loss \
  --encoder-layer-type-list '["conv", "conv", "conv", "conv", "conv", "conv", "conv"]' \
  --encoder-conv-use-glu-list '[1, 1, 1, 1, 1, 1, 1]' \
  --encoder-conv-type-list '["dynamic", "dynamic", "dynamic", "dynamic", "dynamic", "dynamic", "dynamic"]' \
  --encoder-kernel-size-list '[31, 31, 31, 31, 31, 31, 31]' \
  --decoder-layer-type-list '["conv", "conv", "conv", "conv", "conv", "conv"]' \
  --decoder-kernel-size-list '[31, 31, 31, 31, 31, 31]' \
  --decoder-conv-type-list '["dynamic", "dynamic", "dynamic", "dynamic", "dynamic", "dynamic"]' \
  --decoder-conv-use-glu-list '[1, 1, 1, 1, 1, 1]' 
```
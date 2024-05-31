# Keras-based transformer for transliteration.

Instructions for using the transformer code to replicate paper results below.
The instructions here were tested on a linux machine with 64gb ram and 10 CPU
cores. Code is provided as-is and we cannot guarantee support in your
environment. Modify as needed.

## Setup

Using Anaconda Python:

```shell
# Create a virtual environment.
conda create -n clpaper python=3.9
conda activate clpaper
# Install prerequisites.
pip install tf-models-official==2.13.1
```

The 2.13.1 version of `tf-models-official` is the latest known version
compatible with the code here. Later revisions introduce breaking changes that
will require modification.

## Inference

Inference can be performed with the command below:

```shell
python transformer_main.py \
--model_dir=${CHECKPOINT} \
--vocab_file=${VOCAB} \
--max_length=64 \
--param_set=tlit \
--mode=predict_topk \
--beam_size_override=8 \
--infer_file=${INFILE} \
--num_gpus=0 \
--preds_file=${OUTFILE}
```

`CHECKPOINT` should be a path to a checkpoint. Generally, a checkpoint will
consist of two files: a *ckpt.data* and a *ckpt.index* file. You only need
to specify the path up to *.ckpt*.

`VOCAB` should be a path to a vocabulary file.

`INFILE` should be a path to the file to use as input. It should be a text file
in one-line-per-input-word format.

`OUTFILE` should be a path to a TSV file where the inference output will be 
written. The output format includes 3 columns: input word, output prediction,
and likelihood. For example:

```text
adikkum அடிக்கும் -0.17351153
adikkum அதிக்கும் -4.0013576
adikkum ஆடிக்கும் -4.337666
adikkum அடிக்கும்ஊம்       -4.511404
adikkum அடைக்கும் -4.5802116
adikkum அடிக்கும்ொம்       -4.601365
```



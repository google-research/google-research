# Deep Representation One-class Classification (DROC).

[This is not an officially supported Google product.]

This directory contains a two-stage framework for deep one-class classification
example, which includes the self-supervised deep representation learning from
one-class data, and a classifier using generative or discriminative models.

## Install

The requirements.txt includes all the dependencies for this project, and an
example of install and run the project is given in run.sh.

```bash
$sh deep_representation_one_class/run.sh
```

## Download datasets

`script/prepare_data.sh` includes an instruction how to prepare data for
CatVsDog and CelebA datasets. For CatVsDog dataset, the data needs to be
downloaded manually. Please uncomment line 2 to set DATA_DIR to download
datasets before starting it.

## Run

The options for the experiments are specified thru the command line arguments.
The detailed explanation can be found in train_and_eval_loop.py. Scripts for
running experiments can be found

-   Rotation prediction: `script/run_rotation.sh`

-   Contrastive learning: `script/run_contrastive.sh`

-   Contrastive learning with distribution augmentation:
    `script/run_contrastive_da.sh`

## Evaluation

After running train_and_eval_loop.py, the evaluation results can be found in
$MODEL_DIR/stats/summary.json, where MODEL_DIR is specified as model_dir of
train_and_eval_loop.py.

## Contacts

kihyuks@google.com, chunliang@google.com, jinsungyoon@google.com ,
minhojin@google.com, tpfister@google.com

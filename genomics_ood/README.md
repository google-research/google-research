# Out-of-Distribution Detection for Genomics Sequences

This directory contains implementation of the generative model and the classification model for the bacteria genomic dataset that are used in the paper of [Ren J, Liu PJ, Fertig E, Snoek J, Poplin R, DePristo MA, Dillon JV, Lakshminarayanan B. Likelihood Ratios for Out-of-Distribution Detection. arXiv preprint arXiv:1906.02845. 2019 Jun 7.](https://arxiv.org/abs/1906.02845)

## Installation

```bash
virtualenv -p python3 .
source ./bin/activate

pip install -r genomics_ood/requirements.txt
```

## Usage

This directory contains two python scripts:
  generative.py: build an autoregressive generative model for DNA sequences using LSTM.
  classifier.py: build a classifier for DNA sequences using ConvNets.

To test the models on a toy dataset, 

```bash
DATA_DIR=./genomics_ood/test_data
OUT_DIR=./genomics_ood/test_out
python -m genomics_ood.generative \
--hidden_lstm_size=30 \
--val_freq=100 \
--num_steps=1000 \
--in_tr_data_dir=$DATA_DIR/before_2011_in_tr \
--in_val_data_dir=$DATA_DIR/between_2011-2016_in_val \
--ood_val_data_dir=$DATA_DIR/between_2011-2016_ood_val \
--out_dir=$OUT_DIR

python -m genomics_ood.classifier \
--num_motifs=30 \
--val_freq=100 \
--num_steps=1000 \
--in_tr_data_dir=$DATA_DIR/before_2011_in_tr \
--in_val_data_dir=$DATA_DIR/between_2011-2016_in_val \
--ood_val_data_dir=$DATA_DIR/between_2011-2016_ood_val \
--label_dict_file=$DATA_DIR/label_dict.json \
--out_dir=$OUT_DIR
```

## Real Bacteria Dataset

The real bacteria dataset with 10 in-distribtution classes, 60 validation out-of-distribution (OOD) classes, and 60 test OOD classes can be downloaded at [Google Drive](https://drive.google.com/corp/drive/folders/1Ht9xmzyYPbDouUTl_KQdLTJQYX2CuclR)

To run models on the real dataset, one needs to set DATA_DIR=/<path to the real data directory>/ and specify the OUT_DIR.


## Likelihood Ratios

To compute likelihood ratios, we train two generative models using the generative.py. The full model is trained with L2 regularization weight and mutation rate both 0.0. The background model is trained with L2 regularization weight 0.0001 and mutation rate 0.1.



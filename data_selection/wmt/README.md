# Data selection for machine translation

This repo contains code for experiments on data selection for machine translation.
The model, data and experiments are described in our paper (https://arxiv.org/abs/2109.07591).
The main focus of this repo is to explore various tradeoffs and
interactions between data selection and finetuning on out-of-domain and in domain data.
The code is written in python / flax. The model is a vanilla transformer.
The data used is from tfds.datasets. We use the WMT data; specifically Paracrawl and News Commentary.

## Dependencies

All dependencies are listed in requirements.txt. Models are implemented using
the flax/ jax libraries or Huggingface Transformers. Data is sourced from
Tensorflow Datasets (https://www.tensorflow.org/datasets/api_docs/python/tfds).

## Files

The main runner is train.py. This is an example below. There are two helper runners;
clf_infer.py and compute_is.py. Both are to compute selection scores using either
the Descriminative Classifier (DC) or Constrastive Data Selection (CDS) respectively.

## Example

python train.py -- model_dir=models/ --dataset_name='newscommentary_paracrawl' \
  --aux_eval_dataset='newscomment_eval_ft' \
  --batch_size=128 --num_train_steps=15000 \
  --emb_dim=512 --mlp_dim=2048 --num_heads=8 --paracrawl_size=4500000 \
  --vocab_path 'tokenizer/sentencepiece_model' --restore_checkpoints \
  --data_dir='data/' --chkpts_to_keep=1 \
  --checkpoint_freq=5000 --eval_frequency=100 \
  --pretrained_model_dir='pretrained_models/' --save_checkpoints=False \
  --is_scores_path='scores/scores.csv' --data_selection_size=5e5 --compute_bleu=False

Note: If there is no tokenizer, one will be created. data_dir must be populated.
You can download and preprare the data using TF dataset builder (https://www.tensorflow.org/datasets/api_docs/python/tfds/core/DatasetBuilder).
The scores.csv file is a file of the selection scores for each example in the dataset.


## Citations

@article{iter2021complementarity,
  title={On the Complementarity of Data Selection and Fine Tuning for Domain Adaptation},
  author={Iter, Dan and Grangier, David},
  journal={arXiv preprint arXiv:2109.07591},
  year={2021},
  url={https://arxiv.org/abs/2109.07591}
}

This code branches from the Flax WMT example:
https://github.com/google/flax/tree/master/examples/wmt

@software{flax2020github,
  author = {Jonathan Heek and Anselm Levskaya and Avital Oliver and Marvin Ritter and Bertrand Rondepierre and Andreas Steiner and Marc van {Z}ee},
  title = {{F}lax: A neural network library and ecosystem for {JAX}},
  url = {http://github.com/google/flax},
  version = {0.3.4},
  year = {2020},
}

# Latent Programmer

## Overview

Code for Latent Programmer as described in the ICML 2021 paper:
\"[**Latent Programmer: Discrete Latent Codes for Program Synthesis**](https://arxiv.org/abs/2012.00377)"

If you use this codebase, please cite:

```
@article{latentprogrammer,
  title={Latent Programmer: Discrete Latent Codes for Program Synthesis},
  author={Joey Hong and
          David Dohan and
          Rishabh Singh and
          Charles Sutton and
          Manzil Zaheer},
  journal={ICML},
  year={2021}
}
```

## Training

To train a baseline Transformer model, run the following command:

```
python train.py
--logtostderr --dataset_filepattern=[Regex for TFRecord Dataset]
--save_dir=[Path to directory to store results] --num_train_steps=1000000
--num_eval_steps=100
```

To train a Latent Programmer model, run the following:

```
python train_latent.py
--logtostderr --dataset_filepattern=[Regex for TFRecord Dataset]
--save_dir=[Path to directory to store results] --num_train_steps=1000000
--num_eval_steps=100 --latent_vocab_size=30 --num_pretrain_steps=10000
```

Optional flags can be found in `train.py`.

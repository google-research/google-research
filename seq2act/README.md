# Seq2act: Mapping Natural Language Instructions to Mobile UI Action Sequences
This repository contains the code for the models and the experimental framework for "Mapping Natural Language Instructions to Mobile UI Action Sequences" by Yang Li, Jiacong He, Xin Zhou, Yuan Zhang, and Jason Baldridge, which is accepted in 2020 Annual Conference of the Association for Computational Linguistics (ACL 2020).

## Datasets

The data pipelines will be available in future updates.

## Setup

Install the packages that required by our codebase, and perform a test over the setup by running a minimal verion of the model and the experimental framework.

```
sh seq2act/run.sh
```

## Run Experiments.

* Train (and continuously evaluate) seq2act Phrase Tuple Extraction models.

```
sh seq2act/bin/train_seq2act.sh --experiment_dir=your_exp_dir --train=parse --hparam_file=./seq2act/ckpt_hparams/tuple_extract
```

* Train (and continuously evaluate) seq2act Grounding models.

```
sh seq2act/bin/train_seq2act.sh --experiment_dir=your_exp_dir --train=grou nd --hparam_file=./seq2act/ckpt_hparams/grounding
```

* Test the grounding model or only the phrase extraction model by running the decoder.

```
sh seq2act/bin/decode_seq2act.sh
```

If you use any of the materials, please cite the following paper.

```
@inproceedings{seq2act,
  title = {Mapping Natural Language Instructions to Mobile UI Action Sequences},
  author = {Yang Li and Jiacong He and Xin Zhou and Yuan Zhang and Jason Baldridge},
  booktitle = {Annual Conference of the Association for Computational Linguistics (ACL 2020)},
  year = {2020},
  url = {https://arxiv.org/pdf/tbd.pdf},
}
```

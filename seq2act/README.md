# Seq2act: Mapping Natural Language Instructions to Mobile UI Action Sequences
This repository contains the code for the models and the experimental framework for "Mapping Natural Language Instructions to Mobile UI Action Sequences" by Yang Li, Jiacong He, Xin Zhou, Yuan Zhang, and Jason Baldridge, which is accepted in 2020 Annual Conference of the Association for Computational Linguistics (ACL 2020).

## Datasets

Our datasets and data pipelines are released! Please note that we had to 
re-create two of these datasets (AndroidHowTo and PixelHelp) based on public
sources so that they can be opensourced. The re-created datasets led to some 
small differences with the experimental results as those in the paper. Please 
see details of the datasets [here](https://github.com/google-research/google-research/blob/master/seq2act/data_generation/README.md).

## Setup

Install the packages that required by our codebase, and perform a test over the setup by running a minimal verion of the model and the experimental framework.

```
sh seq2act/run.sh
```

## Run Experiments.

* Train (and continuously evaluate) seq2act Phrase Tuple Extraction models.

```
sh seq2act/bin/train_seq2act.sh --experiment_dir=./your_parser_exp_dir --train=parse --hparam_file=./seq2act/ckpt_hparams/tuple_extract
```

Then copy your lastest checkpoint from your_parser_exp_dir to `./seq2act/ckpt_hparams/tuple_extract/`

* Train (and continuously evaluate) seq2act Grounding models.

```
sh seq2act/bin/train_seq2act.sh --experiment_dir=./your_grounding_exp_dir --train=ground --hparam_file=./seq2act/ckpt_hparams/grounding
```

Then copy your latest checkpoint from your_grounding_exp_dir to `./seq2act/ckpt_hparams/grounding/`

NOTE: You can also try out our pre-trained checkpoint for end-to-end grounding
by downloading the checkpoint [here](https://storage.googleapis.com/gresearch/seq2act/ccg3-transformer-6-dot_product_attention-lr_0.003_rd_0.1_ad_0.1_pd_0.2.tar.gz). 
Once downloaded, you can extract the checkpoint files from the zip file, which 
result in 1 file named 'checkpoint' and 3 files with "model.ckpt-250000*".
You can then move these files to  to `./seq2act/ckpt_hparams/grounding/`

* Test the grounding model or only the phrase extraction model by running the decoder.

```
sh seq2act/bin/decode_seq2act.sh --output_dir=./your_decode_dir
```

If you use any of the materials, please cite the following paper.

```
@inproceedings{seq2act,
  title = {Mapping Natural Language Instructions to Mobile UI Action Sequences},
  author = {Yang Li and Jiacong He and Xin Zhou and Yuan Zhang and Jason Baldridge},
  booktitle = {Annual Conference of the Association for Computational Linguistics (ACL 2020)},
  year = {2020},
  url = {https://www.aclweb.org/anthology/2020.acl-main.729.pdf},
}
```

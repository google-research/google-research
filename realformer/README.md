# Residual Attention Layer Transformers (RealFormer)

This repository contains the RealFormer model and pre-trained checkpoints for
"RealFormer: Transformer Likes Residual Attention"
(https://arxiv.org/abs/2012.11747),
published in ACL-IJCNLP 2021.

To cite this work, please use:

```
@inproceedings{he2021realformer,
  title={RealFormer: Transformer Likes Residual Attention},
  author={Ruining He and Anirudh Ravula and Bhargav Kanagal and Joshua Ainslie},
  booktitle={Findings of The Joint Conference of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (ACL-IJCNLP 2021)},
  year={2021}
}
```

## Pre-trained BERT Models with RealFormer

We release pre-trained checkpoints as follows.

Model       | #Layers  | #Heads   | Hidden Size | Intermediate Size | #Parameters | Checkpoint
----------  | :------: | :------: | :------:    | :------:          | :------:    | :------:
BERT-Small  | 4        | 8        | 512         | 2048              | 30M         | [Download](https://storage.googleapis.com/gresearch/realformer/checkpoints/bert_small.zip)
BERT-Base   | 12       | 12       | 768         | 3072              | 110M        | [Donwload](https://storage.googleapis.com/gresearch/realformer/checkpoints/bert_base.zip)
BERT-Large  | 24       | 16       | 1024        | 4096              | 340M        | [Donwload](https://storage.googleapis.com/gresearch/realformer/checkpoints/bert_large.zip)
BERT-xLarge | 36       | 24       | 1536        | 6144              | 1B          | [Donwload](https://storage.googleapis.com/gresearch/realformer/checkpoints/bert_xlarge.zip)

## BERT Fine-tuning

Please follow the standard BERT fine-tuning procedure using the above
pre-trained checkpoints. Hyper-parameter configuration can be found in the
Appendix of the RealFormer paper (https://arxiv.org/abs/2012.11747).


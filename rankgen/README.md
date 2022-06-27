# RankGen

Given an input sequence (or prefix), modern language models often assign high likelihood to output sequences that are repetitive, incoherent, or irrelevant to the prefix; as such, model-generated text also contains such artifacts. To address these issues, we present RankGen, an encoder model (1.2B parameters) that can be flexibly incorporated as a scoring function into beam search and used to decode from any pretrained language model. Experiments across four different language models (345M-11B parameters) and two domains show that RankGen significantly outperforms sampling-based decoding algorithms on both automatic metrics of generation quality (MAUVE) as well as human evaluations. 

## Checkpoints
The following table contains pretrained RankGen checkpoints. We release 6 models in total. Specifically, we release pretrained Base, Large, and XLarge T5 models trained on either the combination of PG19, Wikipedia, C4 news, C4 webtext data ("All") or PG19 data ('PG19")'.

|    Size     | Step | Training Data  | Checkpoint  |
|:-----------:|:----:|:--------------:|:-----------:|
| Base  | 100000| All |  [base-all](https://storage.googleapis.com/gresearch/rankgen/rankgen-base-all.zip)    |
| L     | 100000| All |  [large-all](https://storage.googleapis.com/gresearch/rankgen/rankgen-large-all.zip)   |
| XL    | 100000| All |  [xlarge-all](https://storage.googleapis.com/gresearch/rankgen/rankgen-xlarge-all.zip)  |
| Base  | 100000| PG19|  [base-pg19](https://storage.googleapis.com/gresearch/rankgen/rankgen-base-pg19.zip)   |
| L     | 100000| PG19|  [large-pg19](https://storage.googleapis.com/gresearch/rankgen/rankgen-large-pg19.zip)  |
| XL    | 100000| PG19|  [xlarge-pg19](https://storage.googleapis.com/gresearch/rankgen/rankgen-xlarge-pg19.zip) |

## Usage
See https://github.com/martiansideofthemoon/rankgen.

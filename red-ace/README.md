# RED-ACE
Data and code for the paper [RED-ACE: Robust Error Detection for ASR using Confidence Embeddings](https://aclanthology.org/2022.emnlp-main.180/) (EMNLP 2022).

RED-ACE is an ASR Error Detection (AED) model.
Our approach is based on a modified BERT encoder with an additional embedding
layer, which jointly encodes the textual input and the word-level confidence
scores into a contextualized representation.

This is not an officially supported Google product.

## Code
Please follow `run.sh` for an example how to run the code.

## Data
Our dataset is placed in a public Google Cloud Storage Bucket and can be downloaded from
this [link](https://storage.googleapis.com/gresearch/red-ace/data.zip).

Additional details and data description can be found in the paper.

## BibTeX
If you find this useful for your work, please use the following citation:

```
@misc{2203.07172,
Author = {Zorik Gekhman and Dina Zverinski and Jonathan Mallinson and Genady Beryozkin},
Title = {RED-ACE: Robust Error Detection for ASR using Confidence Embeddings},
Year = {2022},
Eprint = {arXiv:2203.07172},
}
```

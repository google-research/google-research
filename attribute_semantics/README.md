Implementation of discovering attribute semantics described in [Discovering Personalized Semantics for Soft Attributes in Recommender Systems using Concept Activation Vectors](https://arxiv.org/abs/2202.02830).

## Disclaimer

This is not an officially supported Google product.

## Input Data and Sample Usage

For testing with [the MovieLens 20M dataset](https://grouplens.org/datasets/movielens/20m/), we first we transform all tags to lowercase and filter tag-and-rating data to only include the user-item-tags whose corresponding ratings are at least 4. After this pre-filtering, we split the data into train-test set with a roughly (0.75, 0.25) split. During development, we also put one-third of the train set as our validation set. We release the compressed input data and in the code we can restrict the training data to the top tags in terms of unique tagged movies:
[train1](data/train_tags1.csv.gz), [train2](data/train_tags2.csv.gz), [validation](data/validate_tags.csv.gz), and [test](data/test.csv.gz).

Here is a sample command you could use for training attribute semantics with [the MovieLens data](data/train_tags1.csv) and a model consisting of
[(linear) movie embeddings](https://storage.googleapis.com/gresearch/attribute_semantics/cf_embeddings.npz).

```
python movielens_main.py --input_data=./data/train_tags1.csv --save_dir=/tmp --model_path=./cf_embeddings.npz
```

You could then find the models in /tmp/CAVs.

Please first download [the SoftAttributes dataset](https://github.com/google-research-datasets/soft-attributes/blob/main/soft-attributes.csv) before running [softattributes_main.py](softattributes_main.py). Here is a sample command you could use for training and evaluating attribute semantics with [the SoftAttributes dataset](https://github.com/google-research-datasets/soft-attributes/blob/main/soft-attributes.csv).

```
python softattributes_main.py --softattributes_data=soft-attribute-data.csv --save_dir=/tmp --model_path=./model.npz
```

You could then find the model in /tmp/CAVs and some CSV files in /tmp summarizing the evaluation results.

## Acknowledgements

We thank the authors of the SIGIR 2021 paper [On Interpretation and Measurement of Soft Attributes for
Recommendation](https://research.google/pubs/on-interpretation-and-measurement-of-soft-attributes-for-recommendation/) for kindly sharing [the code](soft_attribute.py) of processing [the SoftAttributes dataset](https://github.com/google-research-datasets/soft-attributes/).

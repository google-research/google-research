# *Reviews2Movielens* Dataset

A new multi-document multi-label dataset created by **joining two other existing datasets:** the _Movies and TV_ subset of [Amazon Reviews](https://nijianmo.github.io/amazon/index.html) [2, 3, 4, 5] and [MovieLens](https://grouplens.org/datasets/movielens/) [1], a rich source of crowdsourced movie tags.
This dataset accompanies a paper entitled _DocEnt: Learning Self-Supervised Entity Representations from Large Document Collections_ ([preprint](http://storage.googleapis.com/gresearch/docent/docent_eacl2021_final_v3.pdf)).

#### Introduction

The key challenge in joining the above datasets was establishing correspondences between their respective movie IDs, which turns out to be a many-to-one mapping.
In particular, each Amazon ID (ASIN) matches a canonical product URL (e.g., `B06XGG4FFD` maps to https://www.amazon.com/dp/B06XGG4FFD).
However, these IDs correspond to specific product editions (typically DVDs) rather than unique titles, causing duplication issues.
Moreover, some ASINs are collections of several titles.

We have identified a subset of high-precision, many-to-one correspondences by applying [Named Entity Recognition](https://cloud.google.com/natural-language/docs/basics\#entity\%20analysis) techniques to both Amazon product titles (incl. release years) and their product pages.

The resulting mapping consists of 130026 unique Amazon IDs and 33099 unique MovieLens IDs.
The mapping accuracy was manually verified to be 97% based on 200 random samples.
Ultimately, the joined dataset contains nearly 8.8 million reviews, significantly more than its IMDB counterpart [6].

#### Citation

Please cite the following paper if you use the data in any way:

```
DocEnt: Learning Self-Supervised Entity Representations from Large Document Collections
Yury Zemlyanskiy, Sudeep Gandhe, Ruining He, Bhargav Kanagal, Anirudh Ravula, Juraj Gottweis, Fei Sha and Ilya Eckstein. EACL 2021.
```

Note that both associated datasets (Amazon Reviews and MovieLens) will also need to be cited.

#### Data: v1

This is the original version of *Reviews2Movielens*, as used in the [DocEnt](http://storage.googleapis.com/gresearch/docent/docent_eacl_2021.pdf) paper. It is also the recommended version to use for reproducibility.
The data covers the union of the  _Movies and TV_ subsets of ASINs from two Amazon Reviews dataset versions: [2014](https://jmcauley.ucsd.edu/data/amazon/) and [2013](http://snap.stanford.edu/data/web-Amazon-links.html), mapped to the [MovieLens Dataset](https://grouplens.org/datasets/movielens/latest/) (the latest version as of 10/2020). The data can be accessed by downloading and unzipping [this archive](http://storage.googleapis.com/gresearch/docent/v1.tar.gz). The archive contents are as follows:

**└── Reviews2Movielens.csv** -- the mapping between Amazon Reviews and MovieLens corpora. Every row contains information for a single Amazon item. The columns are:

- **asin**: a unique Amazon ID (ASIN) for an item. Usually, accessible via “https://amazon.com/dp/\<asin>”, but some of the URLs no longer exist.
- **canon_asin**: If we have identified that two Amazon items correspond to the same movie (e.g., one item is blu ray edition and another is DVD) then one of their asins would be chosen as the canonical asin (canon_asin). Note, that every row is guaranteed to have both asin and canon_asin specified.
- **imdb_id**: IMDB id of the item. Could be missing.
- **movielens_id**: MovieLens ID. Could be missing.
- **name**: Name of a movie. Could be missing.
- **imdb_url** IMDB URL of the item. Could be missing.

**└── vocab.ids.txt** -- contains a set of ~81K movie ID used in the experiments. This set has been generated from Amazon Reviews Movies Corpus by (1) applying asin -> canon_asin mapping, (2) removing reviews shorter than 5 words and movies with less than 5 reviews.

**└── docent_models_full_pretrained.tar.gz** -- pretrained DocEnt-FULL models that can also be used as a basis for fine-tuning (see paper for more details).


#### Data: v2

This latest (2nd) version of *Reviews2Movielens* covers the union of _Movies and TV_ subsets of ASINs from several Amazon Reviews dataset versions: [2018](https://nijianmo.github.io/amazon/index.html), [2014](https://jmcauley.ucsd.edu/data/amazon/) and [2013](http://snap.stanford.edu/data/web-Amazon-links.html), mapped to the [MovieLens 25M Dataset](https://grouplens.org/datasets/movielens/25m/). While this version was made to match the most up-to-date Amazon and MovieLens data sources, it does not come with any additional pretrained models. For reproducibility, use `v1` above.
The data can be accessed by downloading and unzipping [this archive](http://storage.googleapis.com/gresearch/docent/v2.tar.gz). The archive includes:

**└── Reviews2Movielens.csv** -- the mapping between Amazon Reviews and MovieLens corpora. Every row contains information for a single Amazon item. The columns are:

- **asin**: a unique Amazon ID (ASIN) for an item. Usually, accessible via “https://amazon.com/dp/\<asin>”, but some of the URLs no longer exist.
- **canon_asin**: If we have identified that two Amazon items correspond to the same movie (e.g., one item is blu ray edition and another is DVD) then one of their asins would be chosen as the canonical asin (canon_asin). Note, that every row is guaranteed to have both asin and canon_asin specified.
- **imdb_id**: IMDB id of the item. Could be missing.
- **movielens_id**: MovieLens ID. Could be missing.
- **name**: Name of a movie. Could be missing.
- **imdb_url** IMDB URL of the item. Could be missing.


#### Code

In addition to data, we provide a few modifications to BERT code needed to pretrain DocEnt-FULL models. Simply use the Python files in the _bert/_ folder of this repository as a drop-in replacement of the corrsepinding files in the [original BERT repo](https://github.com/google-research/bert).



### References

1. The MovieLens Datasets: History and Context. F. Maxwell Harper and Joseph A. Konstan. 2015. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19. https://doi.org/10.1145/2827872
1. Justifying recommendations using distantly-labeled reviews and fined-grained aspects. Jianmo Ni, Jiacheng Li, Julian McAuley. Empirical Methods in Natural Language Processing (EMNLP), 2019
1. Ups and downs: Modeling the visual evolution of fashion trends with one-class collaborative filtering. R. He, J. McAuley. WWW, 2016
1. Image-based recommendations on styles and substitutes. J. McAuley, C. Targett, J. Shi, A. van den Hengel. SIGIR, 2015
1. Hidden factors and hidden topics: understanding rating dimensions with review text. J. McAuley and J. Leskovec. RecSys, 2013
1. Learning Word Vectors for Sentiment Analysis. Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng and Christopher Potts. ACL 2011

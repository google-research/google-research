
# GWikiMatch: A Benchmark Dataset for Long-Form Document Matching

## Dataset Description
GWikiMatch is a benchmark dataset for the long-form document matching task, which
is to estimate the content semantic similarity between document pairs. All
the documents are from English Wikipedia. The creation of this
dataset includes the following steps:

* Candidate document pair generation. We first build a
hyperlink graph based on the linking relations between different Wiki documents.
Then we apply [label propagation](https://research.google/pubs/pub44639/) on the
graph to generate candidate pairs of similar documents. The nodes in the graph
represent documents. There is an undirected edge between two documents if there
is at least one linking relationship between two documents. The existence of an
edge between two nodes indicates that two nodes should end up learning similar
labels. The weights are based on the number of directed hyperlinks between the
two Wikipedia documents. Following the intuition that a document should be
similar to itself, we set the initial label of each node as the URL of the
document. After label propagation, candidate positive document pairs are
generated using documents and their top 100 ranked labels.

* Human annotation on content similarity. The candidate document pairs are
further annotated by humans with content similarity labels. Each
document pair is annotated by 3 human raters on an internal crowdsourcing
platform. We use the majority vote rating
as the final document content similarity label. The rating scale is 0 (not
similar), 1 (somewhat similar) and 2 (strongly similar). In total, we annotated
11K document pairs in 3 different batches. The percentages of document pairs
with label 0/1/2 are 19%/66%/15% respectively.

* Negative sampling. We treat document pairs with label 1 or 2 as the positive
pairs. To generate a more balanced and larger dataset, we adopt negative
sampling to generate the negative pairs following previous
[research](https://research.google/pubs/pub47856/) instead of directly using
document pairs with label 0. Specifically, for each positive
document pair we randomly sample a mismatched document from the outgoing links
of one document in the pair. We check the positive document pairs and
randomly sampled negative pairs to see whether there are any duplicated
document pairs. We remove all the duplicated document pairs and make sure all
the document pairs in the dataset are unique.

* Data partition. We randomly split the document pairs into train/eval/test
partitions with ratio 8:1:1.

* Post-processing to remove controversial topics. We performed further human
annotation to identify documents which contain controversial topics in the data.
We removed document pairs which contain at least one controversial document. We
only release the URL pairs and labels of the remaining data. To demonstrate
how to use the data preprocessing code, we release a small demo data (200
document pairs) in WikiDocPair proto.

## Dataset Statistics

The dataset contains the train/eval/test partitions with human verified
positive document pairs and randomly sampled negative document pairs from
 outgoing links. The dataset statistics are shown in the following table:

| Item                       | Train       | Eval       | Test       |
|----------------------------|-------------|------------|------------|
| # of DocPairs              | 11,758      | 1,435      | 1,468      |


## File Format

The released small demo dataset is in TFRecord files with proto buffer defined by
**WikiDocPair**. We also released the definition of the proto buffer WikiDocPair
in a related code [repository](https://github.com/google-research/google-research/tree/master/smith).
The released train/eval/test datasets which only contain the urls and labels are
in TSV format.

## Download the Data
We uploaded the dataset into Google Cloud Storage. You can download the dataset
following the link [here](http://storage.googleapis.com/gresearch/smith_gwikimatch/README.md).

## Related Code on Data Preprocessing and Model Training

The related code on dataset preprocessing and model training with this dataset
can be found in this
[repository](https://github.com/google-research/google-research/tree/master/smith)
on the model implementation of the SMITH model.


## Licenses
The dataset is licensed by Google LLC under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license.

## Release Notes

- Initial release: 01/11/2021

## Disclaimer

This is not an officially supported Google product.

## Citing

If you use this dataset in your research, please cite the following papers:

```
@inproceedings{Yang2020beyond,
  title={Beyond 512 Tokens: Siamese Multi-depth Transformer-based Hierarchical Encoder for Long-Form Document Matching},
  author={Liu Yang and Mingyang Zhang and Cheng Li and Michael Bendersky and Marc Najork},
  booktitle={CIKM},
  pages = {1725–1734},
  year={2020}
}

@inproceedings{Jiang2019Semantic,
  title={Semantic Text Matching for Long-Form Documents},
  author={Jyun-Yu Jiang and Mingyang Zhang and Cheng Li and Michael Bendersky and Nadav Golbandi and Marc Najork},
  booktitle={WWW},
  pages = {795–806},
  year={2019}
}
```

## Acknowledgements
The following people contributed to the development of this new benchmark
dataset on long-form document matching: Liu Yang, Mingyang Zhang, Cheng Li,
Yi Tay, Tao Chen, Michael Bendersky and Marc Najork.

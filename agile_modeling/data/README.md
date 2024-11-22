# Agile Modeling Domain Expert Data


## About this dataset

This folder contains the annotations labeled by domain experts during the experiments performed in the paper. To facilitate future research, we release all labeled data collected across all experiments, including the ablation studies.

We hope this dataset will assist researchers in further advancing the field of Agile Modeling!

## Dataset structure

For each of the 14 concepts discussed in our paper, the data is structured as follows:

* `<concept_name>_train_labels.csv` contains the union of all training data collected across all ablations of our system for concept `<concept_name>`.
* `<concept_name>_test_labels.csv` contains the test set collected for concept `<concept_name>` using the stratified sampling strategy described in Section 4.2. and detailed in Appendix E of the paper.

Each CSV file consists of two columns:

1. `url`: The url of the image in the LAION-400M dataset that this annotation corresponds to, used to uniquely identify the image in the original dataset.
2. `label`: The label assigned by the domain expert who modeled the corresponding concept, representing the answer to the question "Is this an image of <concept_name>?". This can take the values "positive" or "negative".

## Dataset statistics

Below are the number of examples in each dataset.

| Concept | Train | Test |
|---------|----------------|---------------|
| arts and crafts | 3,001 | 707 |
| astronaut | 3,759 | 637 |
| block tower | 3,942 | 669 |
| dance | 4,051 | 730 |
| emergency service | 6,764 | 675 |
| gourmet tuna | 3,543 | 576 |
| hair-coloring | 4,108 | 645 |
| hand-pointing | 2,811 | 832 |
| healthy-dish | 8,715 | 633 |
| home fragrance | 4,050 | 716 |
| in-ear headphones | 3,687 | 687 |
| pie chart | 2,994 | 594 |
| single sneaker on white background | 4,216 | 556 |
| stop sign | 3,758 | 704 |

## License

The data is available under [CC-BY-4.0 license](https://creativecommons.org/licenses/by/4.0/).

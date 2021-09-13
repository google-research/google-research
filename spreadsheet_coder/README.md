## Introduction

This is a repository for code and data accompanying the ICML 2021 paper
[SPREADSHEETCODER: Formula Prediction from Semi-structured Context](https://arxiv.org/abs/2106.15339).

If you use the code or data released through this repository, please
cite the following paper:
```
@inproceedings{spreadsheetcoder,
author    = {Chen, Xinyun  and
             Maniatis, Petros  and
             Singh, Rishabh  and
             Sutton, Charles and
             Dai, Hanjun and
             Lin, Max and
             Zhou, Denny },
title     = {SPREADSHEETCODER: Formula Prediction from Semi-structured Context},
booktitle = {Proceedings of the 38th International Conference on Machine Learning,
               {ICML} 2021, 18-24 July 2021},
series    = {Proceedings of Machine Learning Research},
publisher = {{PMLR}},
year      = {2021},
}
```

# Code

We are releasing sample code implementing the approach described in the paper.
At this point, the code is provided for illustration purposes, and is not
set up to be executed. Future modifications to this repository may make this
code runnable.


# Datasets

We are releasing a dataset that casts the Enron Spreadsheet corpus to our
representation. The original spreadsheet corpus is on (GitHub)[https://github.com/SheetJS/enron_xls].

You can find the processed files at [[UI]](https://console.cloud.google.com/storage/browser/spreadsheet_coder)
[`gs://spreadsheet_coder`]. The directory contains one subdirectory for examples
extracted from the Enron Spreadsheet corpus. Each file is a shard of all data, maintained in
Tensorflow's TFRecord format. Each record is a tensorflow.train.Example protocol
buffer.

The TensorFlow example features are defined as follows:

* `table_id`: integer, the index of the spreadsheet table.
* `doc_id`: byte string, the index of the spreadsheet file (optional, can be empty).
* `record_index`: integer, the row index of the cell containing the spreadsheet formula.
* `col_index`: integer, the column index of the cell containing the spreadsheet formula.
* `formula`: byte string, the prefix representation of the parse tree of the spreadsheet formula, excluding the cell ranges.
* `formula_token_list`: byte string, the spreadsheet formula sketch, excluding the cell ranges.
* `ranges`: byte string, the cell ranges included in the spreadsheet formula.
* `computed_value`: byte string, the computed value of the spreadsheet formula.
* `header`: byte string, the table header of the column containing the spreadsheet formula.
* `context header`: byte string, the table headers of the surrounding columns with indices in [col_index - 10, col_index + 10].
* `context_data`: byte string, the data of cells that are within 10 rows and 10 columns away from the cell (row_index, col_index).



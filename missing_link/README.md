# The missing link: Finding label relations across datasets

This directory contains the annotated relations between labels across
datasets used in our [ECCV 2022 paper](https://arxiv.org/abs/2206.04453).
The annotations were created by building on existing annotations of 
the [MSEG dataset](https://github.com/mseg-dataset/mseg-api)
([Lambert et al. CVPR 2020](https://arxiv.org/abs/2112.13762)).
We used this to automatically establish relationships which we subsequently
manually inspected and manually corrected where necessary.

We provide label relations across datasets between three dataset pairs:

* ADE20k and COCO: `ade_coco_mapping.csv`
* ADE20k and BDD: `ade_bdd_mapping.csv`
* BDD and COCO: `bdd_coco_mapping.csv`

For completeness, we also include the mappings to the MSEG dataset
which we used as an intermediate step to produce these label relations.

The original datasets and papers can be found at:

* [ADE20k dataset](https://groups.csail.mit.edu/vision/datasets/ADE20K) ([Zhou et al. IJCV 2017](https://arxiv.org/abs/1608.05442))
* [Berkeley Deep Drive (BDD) dataset](https://www.bdd100k.com/) ([Yu et al. CVPR 2020](https://arxiv.org/abs/1805.04687))
* [COCO panoptic dataset](https://cocodataset.org/#home) ([Lin et al. ECCV 2014](https://arxiv.org/abs/1405.0312), [Caesar et al. CVPR 2018](https://arxiv.org/abs/1612.03716), [Kirillov et al. CVPR 2019](https://arxiv.org/abs/1801.00868))

## File Format

All files are csv files whose rows have the following format:

```
label_a, relation, label_b
```

If the filename is ade_bdd_mapping.csv, then `label_a` refers to a label in `ade`
and `label_b` refers to a label in `bdd`. The `relation` is one of:

* **identity**: `label_a` and `label_b` represent the same visual concept.
* **parent**: `label_a` is a parent of `label_b`.
* **child**: `label_a` is a child of `label_b`.
* **overlap**: `label_a` and `label_b` describe visual concepts which are
similar but not the same: their sets of instances intersect, but each set
contain instances which are not present in the other set.
* **has_part**: `label_a` has a part `label_b` (unused in our paper).
* **part_of**: `label_a` is a part of `label_b` (unused in our paper).
* **sibling**: `label_a` is disjoint with `label_b` but share a parent (incomplete, unused in our paper).

The relations are reversible but may change as is obvious from their semantics
(e.g. if `a` is a `parent` of `b`, then `b` is a `child` of `a`).
Note that the csv files contain some comments which provide a rationale for the
specific relationship type. To properly load a file one can use:

```python
import csv

def remove_comments(csvfile):
  for row in csvfile:
    row_and_comment = row.split('#')
    row_without_comment = row_and_comment[0].strip()
    if row_without_comment:
      yield row_without_comment

with open(filename) as f:
  csv_reader = csv.reader(remove_comments(f))
  for item in csv_reader:
    # process item, which has form [label_a, relation, label_b]
```

## Citation

If you use our annotations please cite our work:

```
@inproceedings{uijlings22eccv,
  title={The Missing Link: Finding label relations across datasets},
  author={Jasper Uijlings and Thomas Mensink and Vittorio Ferrari},
  year={2022},
  booktitle={ECCV},
}
```


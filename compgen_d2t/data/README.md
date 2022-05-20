## FewShot SGD

Please find the original dataset in [here](https://github.com/google-research/schema-guided-dialogue/tree/main/generation).

## FewShot Weather

The data is put under:
```
$ gs://scenic-bucket/acl2022-compgen/
```


The [weather dataset](https://github.com/facebookresearch/TreeNLG/tree/master/data/weather) is originally collected from:

Anusha Balakrishnan, Jinfeng Rao, Kartikeya Upasani, Michael White and Rajen Subba. [Constrained Decoding for Neural NLG from Compositional Representations in Task-Oriented Dialogue](https://www.aclweb.org/anthology/P19-1080/), ACL 2019.

The original dataset contains 8 total distinct discourse relations and dialog acts, and 47 unique arguments. It contains a total of 4,690 unique meaning represetantions (aka. unique tree structures). The original train/val/test statistics are shown below:

Split   | #Examples | #Unique Structures
------- | ----- | ---
train | 25390 | 4690
val | 3078 | 1178
test | 3121 | 1217

Please consider citing the original paper if you are using our data variant.

### FewShot Split Statsitics
Each file in this repo contains 5 fields, fome left to right: `ID`, `User Query`, `Meaning Representation (MR)`, `Reference` and `Template Output`. The `User Query` field concatenated with the `Template Output` field by a `[SEP]` token is used as the input for T5 models, and `Reference` is used as the target for T5 models during training and evaluation.

The few-shot splits are sampled from the original `train` split, with X-shot denotes X unique structures with one example per structure. The `delex_full` split contains the left examples from the original `train` split that have different structrues as `delex_train_1000`. The statistics of all training splits are shown below:

Split           | #Examples | #Unique Structures
-------         | --------- | ----------
delex_train_250.tsv         | 250 | 250
delex_train_500.tsv         | 500 | 500
delex_train_750.tsv         | 750 | 750
delex_train_1000.tsv        | 1000 | 1000
delex_full_train.tsv        | 16816 | N/A

For all training splits, the models are validated and evaluated on both seen and unseen structures. The seen val/test splits are selected from the original `val/test` splits by filtering examples that have unseen structures as `delex_train_1000`, and vice versa for the unseen splits.

Split           | #Examples | #Unique Structures
-------         | --------- | ----------
delex_val_seen.tsv         | 1087 | N/A
delex_test_seen.tsv        | 1121 | N/A
delex_val_unseen.tsv         | 1095 | N/A
delex_test_unseen.tsv        | 1170 | N/A

For the self-training experiments, we constructed an `unlabelled` set from the `train` set by removing the examples in `delex_train_1000`. The statistics can be found as below:

Split           | #Examples | #Unique Structures
-------         | --------- | ----------
delex_unlabeled.tsv | 24390 | N/A
delex_unlabeled_seen_structs.tsv | N/A | N/A
delex_unlabeled_unseen_structs.tsv | N/A | N/A

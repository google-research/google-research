# CFQ classification task

This repository contains the source code for the paper: https://arxiv.org/abs/2106.10434

```
@misc{kim2021improving,
      title={Improving Compositional Generalization in Classification Tasks via Structure Annotations},
      author={Juyong Kim and Pradeep Ravikumar and Joshua Ainslie and Santiago Ontañón},
      year={2021},
      eprint={2106.10434},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

The code converts the CFQ dataset into a sentence pair classification task and trains ML models on the task annotated with structural annotation.

## 1. Create datasets
1. Place CFQ dataset under `scripts` directory. A file `dataset.json` of CFQ dataset should be placed in `scripts/cfq`.
2. (For model negative dataset) Run CFQ baseline code with option to print beam score (Open `run.sh` and append `--decode_hparams="return_beams=True,write_beam_scores=True"` to `t2t-decoder` command). Place CFQ baseline outputs under `scripts/cfq_model_outputs` directory. The directory structure should be
```
scripts
└─cfq_model_outputs
  └─(cfq_split)
    ├─train_encode.txt
    ├─train_decode_lstm.txt
    ├─train_decode_transformer.txt
    ├─train_decode_universal.txt
    ├─dev_encode.txt
    ...
    ├─dev_decode_universal.txt
    ├─test_encode.txt
    ...
    └─test_decode_universal.txt
```
3. Run the generation shell script in `scripts` (Please see help output for usage).
```
$ ./create_cls_dataset.sh cfq_split neg_method train_hold_out output_tree
```
**Note**: Currently, dataset with structure annotation (or when `output_tree` is `true`) can be generated when `xlink_mapping.pkl` is placed under the dataset output dir (This file can be generated using a jupyter notebook `colab/cfq_xlink_mutual_information.ipynb` and the dataset of the same config but without structure annotation).

## 2. Dependency
Checkout the ETC repository at (https://github.com/google-research/google-research/tree/master/etcmodel) under the `third_party` directory.

## 3. Run model
Run one of the training shell scripts (`cls_*.sh`) in the project root (Please see help output for usage). Note that only Relative Transformer can use structure annotations.

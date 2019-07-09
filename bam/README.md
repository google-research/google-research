# Born-Again Multi-task Networks

This directory contains code for [BAM! Born-Again Multi-task for Natural Language Understanding](https://openreview.net/forum?id=SylnYlqKw4). It supports TPU-friendly multi-task fine-tuning of [BERT](https://arxiv.org/abs/1810.04805) (optionally) combined with knowledge distillation.

## Requirements
See `requirements.txt`

## Setup
1. Create a directory `BAM_DIR` to contain training data, model checkpoints, etc.
2. Download pre-trained BERT weights from [here](https://github.com/google-research/bert) and unzip them under `$BAM_DIR/pretrained_models`. By default, the uncased base-sized model is expected (i.e., you should have .
3. Download the GLUE data by running [this script](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e). Set up the data by running `mv CoLA cola && mv MNLI mnli && mv MRPC mrpc && mv QNLI qnli && mv QQP qqp && mv RTE rte && mv SST-2 sst && mv STS-B sts && mv diagnostic/diagnostic.tsv mnli && mkdir -p $BAM_DIR/glue_data && mv * $BAM_DIR/glue_data`.

## Training and evaluating models
Run training/evaluation with `python -m bam.run_classifier <model_name> <bam_dir> <hparams>` where `<hparams>` is [JSON](https://www.json.org/) containing hyperparameters, which tasks to train on, etc. See `configure.py` for details on the available options.
Here are some example commands (run from the top-level `google_research` directory):
* Debug run (trains a tiny model, takes around a minute): `python -m bam.run_classifier debug-model $BAM_DIR '{"debug": true}'`
* Train single-task models on the RTE and MRPC datasets: `python -m bam.run_classifier rte-model $BAM_DIR '{"task_names": ["rte"]}'`
and `python -m run_classifier mrpc-model $BAM_DIR '{"task_names": ["mrpc"]}'`
* Train a multi-task model on the RTE and MRPC datasets: `python -m bam.run_classifier rte-mrpc-model $BAM_DIR '{"task_names": ["rte", "mrpc"]}'`
* Train a BAM model for the RTE and MRPC datasets (requires that single-task models have been trained):
`python -m bam.run_classifier rte-mrpc-bam-model $BAM_DIR '{"task_names": ["rte", "mrpc"], "distill": true, "teachers": {"rte": "rte-model", "mrpc": "mrpc-model"}}'`

Training a large model on GPU may cause out-of-memory issues, in which case you can decrease the batch size/learning rate, e.g:
* `python -m bam.run_classifier rte-mrpc-bam-model $BAM_DIR '{"task_names": ["rte", "mrpc"], "distill": true, "train_batch_size": 64, "learning_rate": 5e-5, "teachers": {"rte": "rte-model", "mrpc": "mrpc-model"}}'`


## Citation
If you use this code for your publication, please cite the original paper:
```
@inproceedings{clark2019bam,
  title = {BAM! Born-Again Multi-Task Networks for Natural Language Understanding},
  author = {Kevin Clark and Minh-Thang Luong and Christopher D. Manning and Quoc V. Le},
  booktitle = {ACL},
  year = {2019}
}
```

## Contact
* [Kevin Clark](https://cs.stanford.edu/~kevclark/) (@clarkkev).

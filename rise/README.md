## RISE: Retrieval-Inspired Summarization Evaluation

This repository contains information about evaluating summaries and the
required checkpoints.

The associated paper can be found at: https://arxiv.org/abs/2212.08775

### Evaluating

RISE is built on top of
[T5X Retrieval](https://github.com/google-research/t5x_retrieval).
The easiest way to the evaluate a model and its associated outputs (given a
set of input documents to summarize) is to create a
[SeqIO](https://github.com/google/seqio)
task and run inference upon that task using
[T5X Infer](https://github.com/google-research/t5x#inference).

An example run, assuming one has set up their dataset as a SeqIO task, and
making use of one of our T5.1.1 checkpoints:

```
INFER_OUTPUT_DIR="..."  # directory to write infer output
T5X_DIR="..."  # directory where the t5x is cloned, e.g., ${HOME}"/t5x".
CHECKPOINT_PATH="..."

python3 ${T5X_DIR}/t5x/infer.py \
  --gin_file=t5x_retrieval/configs/runs/infer.gin \
  --gin_file=t5x_retrieval/configs/models/de_t5_1_1_large.gin \
  --gin.CHECKPOINT_PATH=\"${CHECKPOINT_PATH}\" \
  --gin.INFER_OUTPUT_DIR=\"${INFER_OUTPUT_DIR}\" \
  --gin.infer.mode="'score'" \
  --gin.TASK_FEATURE_LENGTHS="${FLAGS_task_feature_lengths}" \
  --gin.T5XR_INFERENCE_MODE="'similarity'" \
  --gin.TASK_FEATURE_LENGTHS="{'inputs': 4096, 'targets': 512}"
```

### Checkpoints

The associated checkpoints can be found at
[this link](https://console.cloud.google.com/storage/browser/gresearch/rise).

As described in the paper, there are three variants of checkpoints depending
on what base model they were trained (
[T5.1.1](https://github.com/google-research/text-to-text-transfer-transformer/blob/main/released_checkpoints.md),
[LongT5](https://github.com/google-research/longt5),
or [mT5](https://github.com/google-research/multilingual-t5)).
Additionally, for each dataset we finetuned upon, we are also releasing both
a model trained only on lexical negatives, and a model trained on a combination
of lexical and model negatives.

### Citing RISE

Please use the following bibtex entry to cite RISE.

```
@article{uthus2022rise,
  url = {https://arxiv.org/abs/2212.08775},
  author = {Uthus, David and Ni, Jianmo},
  title = {RISE: Leveraging Retrieval Techniques for Summarization Evaluation},
  journal={arXiv preprint arXiv:2212.08775},
  year = {2022},
}
```

### Disclaimer

This is not an official Google product.

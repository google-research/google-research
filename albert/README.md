ALBERT
======

ALBERT is "A Lite" version of BERT, a popular unsupervised language
representation learning algorithm. ALBERT uses parameter-reduction techniques
that allow for large-scale configurations, overcome previous memory limitations,
and achieve better behavior with respect to model degradation.

For a technical description of the algorithm, see our paper:

[ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942)

Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut

Release Notes
=============

- Initial release: 10/9/2019

Results
=======

Performance of ALBERT on GLUE benchmark results using a single-model setup on
dev:

| Models            | MNLI     | QNLI     | QQP      | RTE      | SST      | MRPC     | CoLA     | STS      |
|-------------------|----------|----------|----------|----------|----------|----------|----------|----------|
| BERT-large        | 86.6     | 92.3     | 91.3     | 70.4     | 93.2     | 88.0     | 60.6     | 90.0     |
| XLNet-large       | 89.8     | 93.9     | 91.8     | 83.8     | 95.6     | 89.2     | 63.6     | 91.8     |
| RoBERTa-large     | 90.2     | 94.7     | **92.2** | 86.6     | 96.4     | **90.9** | 68.0     | 92.4     |
| ALBERT (1M)       | 90.4     | 95.2     | 92.0     | 88.1     | 96.8     | 90.2     | 68.7     | 92.7     |
| ALBERT (1.5M)     | **90.8** | **95.3** | **92.2** | **89.2** | **96.9** | **90.9** | **71.4** | **93.0** |

Performance of ALBERT-xxl on SQuaD and RACE benchmarks using a single-model
setup:

|Models                    | SQuAD1.1 dev  | SQuAD2.0 dev  | SQuAD2.0 test | RACE test (Middle/High) |
|--------------------------|---------------|---------------|---------------|-------------------------|
|BERT-large                | 90.9/84.1     | 81.8/79.0     | 89.1/86.3     | 72.0 (79.6/70.1)        |
|XLNet                     | 94.5/89.0     | 88.8/86.1     | 89.1/86.3     | 81.8 (85.5/80.2)        |
|RoBERTa                   | 94.6/88.9     | 89.4/86.5     | 89.8/86.8     | 83.2 (86.5/81.3)        |
|UPM                       | -             | -             | 89.9/87.2     | -                       |
|XLNet + SG-Net Verifier++ | -             | -             | 90.1/87.2     | -                       |
|ALBERT (1M)               | 94.8/89.2     | 89.9/87.2     | -             | 86.0 (88.2/85.1)        |
|ALBERT (1.5M)             | **94.8/89.3** | **90.2/87.4** | **90.9/88.1** | **86.5 (89.0/85.5)**    |


Pre-trained Models
==================
TF-Hub modules are available:
- [https://tfhub.dev/google/albert_base/1](albert_base).
- [https://tfhub.dev/google/albert_large/1](albert_large)
- [https://tfhub.dev/google/albert_xlarge/1](albert_xlarge)
- [https://tfhub.dev/google/albert_xxlarge/1](albert_xxlarge)

Example usage of the TF-Hub module:

```
tags = set()
if is_training:
  tags.add("train")
albert_module = hub.Module("https://tfhub.dev/google/albert_base/1", tags=tags,
                           trainable=True)
albert_inputs = dict(
    input_ids=input_ids,
    input_mask=input_mask,
    segment_ids=segment_ids)
albert_outputs = albert_module(
    inputs=albert_inputs,
    signature="tokens",
    as_dict=True)

# If you want to use the token-level output, use
# albert_outputs["sequence_output"] instead.
output_layer = albert_outputs["pooled_output"]
```

For a full example, see `run_classifier_with_tfhub.py`.

Pre-training Instructions
=========================
Use `run_pretraining.py` to pretrain ALBERT:

```
pip install -r albert/requirements.txt
python -m albert.run_pretraining \
    --output_dir="${OUTPUT_DIR}" \
    --export_dir="${EXPORT_DIR}" \
    --do_train \
    --do_eval \
    <additional flags>
```

Fine-tuning Instructions
========================
For XNLI, COLA, MNLI, and MRPC, use `run_classifier_sp.py`:

```
pip install -r albert/requirements.txt
python -m albert.run_classifier_sp \
  --task_name=MNLI \
  <additional flags>
```

You should see some output like this:

```
***** Eval results *****
  global_step = ...
  loss = ...
  masked_lm_accuracy = ...
  masked_lm_loss = ...
  sentence_order_accuracy = ...
  sentence_order_loss = ...
```

You can also fine-tune the model starting from TF-Hub modules using
`run_classifier_with_tfhub.py`:

```
pip install -r albert/requirements.txt
python -m albert.run_classifier_with_tfhub \
  --albert_hub_module_handle=https://tfhub.dev/google/albert_base/1 \
  <additional flags>
```

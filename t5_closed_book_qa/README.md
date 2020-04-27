# Closed Book Question Answering w/ T5

This repository contains the code for reproducing the experiments in
[How Much Knowledge Can You Pack Into the Parameters of a Language Model?](https://arxiv.org/abs/2002.08910).

## Table of Contents

* [Usage](#usage)
* [Released Model Checkpoints](#released-model-checkpoints)
* [How to Cite](#how-to-cite)

## Usage

To run this code, you first need to install the
[t5 library](https://pypi.org/project/t5/). General instructions for training, fine-tuning, evaluation, and exporting models for inference can be found in the [t5 repo](https://github.com/google-research/text-to-text-transfer-transformer).

In order to use the additional CBQA tasks provided in this library with the `t5_mesh_transformer` commands, run from this directory and add the flag `--module_import="tasks"`.
If using the `t5` API from an interactive shell or script, simply call `import tasks`.

As an example, you can fine-tune on a mixture of all 3 CBQA tasks
(Natural Questions, Web Questions, and TriviaQA) with the
T5-11B model by running the command below from this directory.

The remaining experiments are shown in the [tasks.py](tasks.py) file.

```
export PROJECT=yourproject
export ZONE=yourzone
export BUCKET=yourbucket
export TPU=yourtpu

ctpu up   --name=$TPU   --project=$PROJECT  --zone=$ZONE   --tpu-size=v3-128   --tpu-only   --noconf

TASK=closed_book_qa
PRETRAINED_DIR=gs://t5-data/pretrained_models/11B
PRETRAINED_STEPS=1000000

# Run fine-tuning
t5_mesh_transformer \
  --module_import="tasks" \
  --tpu="${TPU}" \
  --gcp_project="${PROJECT}" \
  --tpu_zone="${ZONE}" \
  --model_dir="${BUCKET}/{$TASK}/11B/" \
  --gin_file="dataset.gin" \
  --gin_file="${PRETRAINED_DIR}/operative_config.gin" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = '8x8'" \
  --gin_param="MIXTURE_NAME = '${TASK}'" \
  --gin_param="mesh_train_dataset_fn.use_cached=False" \
  --gin_param="utils.run.save_checkpoints_steps=100" \
  --gin_param="utils.run.batch_size=('tokens_per_batch', 196608)" \
  --gin_param="utils.run.train_steps=1010000" \
  --gin_param="utils.run.init_checkpoint='${PRETRAINED_DIR}/11B/model.ckpt-${PRETRAINED_STEPS}'" \
  --gin_param="utils.run.learning_rate_schedule=@learning_rate_schedules.constant_learning_rate" \
  --gin_param="constant_learning_rate.learning_rate=1e-3" \
  --t5_tfds_data_dir="${BUCKET}/t5-tfds" 

# Run eval
t5_mesh_transformer \
  --module_import="tasks" \
  --tpu="${TPU}" \
  --gcp_project="${PROJECT}" \
  --tpu_zone="${ZONE}" \
  --model_dir="${BUCKET}/${TASK}/11B/" \
  --gin_file="dataset.gin" \
  --gin_file="${BUCKET}/${TASK}/11B/operative_config.gin" \
  --gin_file="eval.gin" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = '8x8'" \
  --gin_param="MIXTURE_NAME = '${TASK}"" \
  --gin_param="mesh_eval_dataset_fn.use_cached=False" \
  --gin_param="utils.run.dataset_split = 'validation'" \
  --gin_param="utils.run.batch_size=('tokens_per_batch', 65536)" \
  --gin_param="utils.run.eval_checkpoint_step='all'" \
  --t5_tfds_data_dir="${BUCKET}/t5-tfds"
```

## Released Model Checkpoints

To facilitate reproducibility and future work, we have released the model checkpoints for our largest (and best-performing) models, which are the most difficult to train.

Each was initialized with a pre-trained T5 checkpoint (available in the
[t5 repo](https://github.com/google-research/text-to-text-transfer-transformer#released-model-checkpoints))
and pre-trained for an additional 100k steps with "salient span masking" (SSM) on the dataset of Wikipedia sentences from [Guu et al., 2020](https://arxiv.org/pdf/2002.08909.pdf).
These models were then fine-tuned with 10k steps on individual open-domain QA tasks. For more details on our training procedure, see [our paper](https://arxiv.org/abs/2002.08910).

* **T5-11B + SSM:** [gs://t5-data/pretrained_models/cbqa/11b_ssm](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/cbqa/11b_ssm)
* **T5-11B + SSM + Natural Questions (train):** [gs://t5-data/pretrained_models/cbqa/11b_ssm_nq](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/cbqa/11b_ssm_nq)
* **T5-11B + SSM + WebQuestions (train):** [gs://t5-data/pretrained_models/cbqa/11b_ssm_wq](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/cbqa/11b_ssm_wq)
* **T5-11B + SSM + TriviaQA (train + validation):** [gs://t5-data/pretrained_models/cbqa/11b_ssm_tqa](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/cbqa/11b_ssm_tqa)
* **T5.1.1-XXL + SSM:** [gs://t5-data/pretrained_models/cbqa/t5.1.1.xxl_ssm](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/cbqa/t5.1.1.xxl_ssm)
* **T5.1.1-XXL + SSM + Natural Questions (train):** [gs://t5-data/pretrained_models/cbqa/t5.1.1.xxl_ssm_nq](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/cbqa/t5.1.1.xxl_ssm_nq)
* **T5.1.1-XXL + SSM + WebQuestions (train):** [gs://t5-data/pretrained_models/cbqa/t5.1.1.xxl_ssm_wq](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/cbqa/t5.1.1.xxl_ssm_wq)
* **T5.1.1-XXL + SSM + TriviaQA (train + validation):** [gs://t5-data/pretrained_models/cbqa/t5.1.1.xxl_ssm_tqa](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/cbqa/t5.1.1.xxl_ssm_tqa)

# How to Cite
If you extend or use this work, please cite the [paper](https://arxiv.org/abs/2002.08910) where it was introduced:

```
@article{2020t5cqba,
  author = {Adam Roberts and Colin Raffel and Noam Shazeer},
  title = {How Much Knowledge Can You Pack Into the Parameters of a Language Model?},
  journal = {arXiv e-prints},
  year = {2020},
  archivePrefix = {arXiv},
  eprint = {2002.08910},
}
```

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

In order to use the additional CBQA tasks provided in this library with the `t5_mesh_transformer` commands, run from this directory and add the flag `--module_import="t5_cbqa.tasks"`.
If using the `t5` API from an interactive shell or script, simply call `import t5_cbqa.tasks`.

As an example, you can fine-tune on a mixture of all 3 CBQA tasks
(Natural Questions, Web Questions, and TriviaQA) with the
T5.1.1-XXL + SSM model by running the command below from this directory.

The remaining experiments are shown in the [tasks.py](t5_cbqa/tasks.py) file.

```shell
PROJECT=yourproject
ZONE=yourzone
BUCKET=gs://yourbucket
TPU=yourtpu
TPU_SIZE=v3-64

ctpu up --name=$TPU --project=$PROJECT --zone=$ZONE --tpu-size=$TPU_SIZE --tpu-only --noconf

TASK=closed_book_qa
PRETRAINED_DIR=gs://t5-data/pretrained_models/cbqa/t5.1.1.xxl_ssm
PRETRAINED_STEPS=1100000
FINETUNE_STEPS=10000
MODEL_DIR="${BUCKET}/${TASK}/xxl_ssm"

# Run fine-tuning
python -m t5.models.mesh_transformer_main \
  --module_import="t5_cbqa.tasks" \
  --tpu="${TPU}" \
  --gcp_project="${PROJECT}" \
  --tpu_zone="${ZONE}" \
  --model_dir="${MODEL_DIR}" \
  --gin_file="dataset.gin" \
  --gin_file="${PRETRAINED_DIR}/operative_config.gin" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = '${TPU_SIZE}'" \
  --gin_param="MIXTURE_NAME = '${TASK}'" \
  --gin_param="utils.run.save_checkpoints_steps=1000" \
  --gin_param="utils.run.batch_size=('tokens_per_batch', 196608)" \
  --gin_param="utils.run.train_steps=$((PRETRAINED_STEPS+FINETUNE_STEPS))" \
  --gin_param="utils.run.init_checkpoint='${PRETRAINED_DIR}/model.ckpt-${PRETRAINED_STEPS}'" \
  --gin_param="utils.run.learning_rate_schedule=@learning_rate_schedules.constant_learning_rate" \
  --gin_param="constant_learning_rate.learning_rate=1e-3" \
  --t5_tfds_data_dir="${BUCKET}/t5-tfds"

# Run eval
python -m t5.models.mesh_transformer_main \
  --module_import="t5_cbqa.tasks" \
  --tpu="${TPU}" \
  --gcp_project="${PROJECT}" \
  --tpu_zone="${ZONE}" \
  --model_dir="${MODEL_DIR}" \
  --gin_file="dataset.gin" \
  --gin_file="${MODEL_DIR}/operative_config.gin" \
  --gin_file="eval.gin" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = '${TPU_SIZE}'" \
  --gin_param="MIXTURE_NAME = '${TASK}'" \
  --gin_param="utils.run.dataset_split = 'validation'" \
  --gin_param="utils.run.batch_size = 128" \
  --gin_param="utils.run.eval_checkpoint_step = 'all'" \
  --t5_tfds_data_dir="${BUCKET}/t5-tfds"
```

## Released Model Checkpoints

To facilitate reproducibility and future work, we have released the model checkpoints for our largest (and best-performing) models, which are the most difficult to train.

Each was initialized with a pre-trained T5 checkpoint (available in the
[t5 repo](https://github.com/google-research/text-to-text-transfer-transformer#released-model-checkpoints))
and pre-trained for an additional 100k steps with "salient span masking" (SSM) on the dataset of Wikipedia sentences from [Guu et al., 2020](https://arxiv.org/pdf/2002.08909.pdf).

The models fine-tuned on `*_open_test` tasks were fine-tuned with 10k steps on individual open-domain QA tasks using the full train splits (and also the validation split in the case of TriviaQA). The released checkpoint is from the final step of fine-tuning.

The models fine-tuned on `*_open` tasks were trained for 20k steps on ~90% of the train split. The released checkpoint is the one producing the best score on the held-out ~10% of the train split.

For more details on our training procedure, see [our paper](https://arxiv.org/abs/2002.08910).


SSM Models with no fine-tuning:

| Base Model | Path |
| :----: | :------------: |
| T5-small | [gs://t5-data/pretrained_models/cbqa/small_ssm](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/cbqa/small_ssm) |
| T5-11B | [gs://t5-data/pretrained_models/cbqa/11b_ssm](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/cbqa/11b_ssm) |
| T5.1.1-XXL | [gs://t5-data/pretrained_models/cbqa/t5.1.1.xxl_ssm](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/cbqa/t5.1.1.xxl_ssm) |

SSM models fine-tuned on Natural Questions:

| Base Model (+SSM) |  Finetune Task | EM Score | Path |
| :----: | :------------: | :-----------: | :------: |
| T5-small | `natural_questions_open_test` | 25.5 | [gs://t5-data/pretrained_models/cbqa/small_ssm_nq](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/cbqa/small_ssm_nq) |
| T5-11B | `natural_questions_open_test` | 36.6  | [gs://t5-data/pretrained_models/cbqa/11b_ssm_nq](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/cbqa/11b_ssm_nq) |
| T5.1.1-XL | `natural_questions_open_test` | 35.6  | [gs://t5-data/pretrained_models/cbqa/t5.1.1.xl_ssm_nq](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/cbqa/t5.1.1.xl_ssm_nq) |
| T5.1.1-XXL | `natural_questions_open_test` | 37.9  | [gs://t5-data/pretrained_models/cbqa/t5.1.1.xxl_ssm_nq](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/cbqa/t5.1.1.xxl_ssm_nq) |
| T5-11B | `natural_questions_open` | 34.8  | [gs://t5-data/pretrained_models/cbqa/11b_ssm_nqo](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/cbqa/11b_ssm_nqo) |
| T5.1.1-XXL | `natural_questions_open` | 35.2  | [gs://t5-data/pretrained_models/cbqa/t5.1.1.xxl_ssm_nqo](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/cbqa/t5.1.1.xxl_ssm_nqo) |

SSM models fine-tuned on WebQuestions:

| Base Model (+SSM) |  Finetune Task | EM Score | Path |
| :----: | :------------: | :-----------: | :------: |
| T5-11B | `web_questions_open_test` | 44.7  | [gs://t5-data/pretrained_models/cbqa/11b_ssm_wq](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/cbqa/11b_ssm_wq) |
| T5.1.1-XXL | `web_questions_open_test` | 43.5  | [gs://t5-data/pretrained_models/cbqa/t5.1.1.xxl_ssm_wq](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/cbqa/t5.1.1.xxl_ssm_wq) |
| T5-11B | `web_questions_open` | 40.8  | [gs://t5-data/pretrained_models/cbqa/11b_ssm_wqo](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/cbqa/11b_ssm_wqo) |
| T5.1.1-XXL | `web_questions_open` | 42.8  | [gs://t5-data/pretrained_models/cbqa/t5.1.1.xxl_ssm_wqo](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/cbqa/t5.1.1.xxl_ssm_wqo) |

SSM models fine-tuned on TriviaQA:

| Base Model (+SSM) |  Finetune Task | EM Score | Path |
| :----: | :------------: | :-----------: | :------: |
| T5-11B | `trivia_qa_open_test` | 60.5✝  | [gs://t5-data/pretrained_models/cbqa/11b_ssm_tqa](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/cbqa/11b_ssm_tqa) |
| T5.1.1-XXL | `trivia_qa_open_test` | 61.6✝ | [gs://t5-data/pretrained_models/cbqa/t5.1.1.xxl_ssm_tqa](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/cbqa/t5.1.1.xxl_ssm_tqa) |
| T5-11B | `trivia_qa_open` | 51.0  | [gs://t5-data/pretrained_models/cbqa/11b_ssm_tqao](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/cbqa/11b_ssm_tqa) |
| T5.1.1-XXL | `trivia_qa_open` | 51.9 | [gs://t5-data/pretrained_models/cbqa/t5.1.1.xxl_ssm_tqao](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/cbqa/t5.1.1.xxl_ssm_tqao) |

✝ Score for the private TriviaQA Wikipedia domain test set.

# How to Cite
If you extend or use this work, please cite the [paper](https://arxiv.org/abs/2002.08910) where it was introduced:

```
@inproceedings{2020t5cqba,
  author = {Adam Roberts and Colin Raffel and Noam Shazeer},
  title = {How Much Knowledge Can You Pack Into the Parameters of a Language Model?},
  booktitle = {Empirical Methods in Natural Language Processing (EMNLP)},
  year = {2020},
}
```

# WT5

This repository contains the code for reproducing the experiments in
"WT5?! Training Text-to-Text Models to Explain their Predictions".

To run this code, you need to install the
[t5 library](https://pypi.org/project/t5/). General instructions for training, fine-tuning, evaluation, and exporting models for inference can be found in the [t5 repo](https://github.com/google-research/text-to-text-transfer-transformer). In order to use the additional WT5 tasks and mixtures provided in this library with the `t5_mesh_transformer` commands, run from this directory and add the flag `--module_import="wt5.mixtures"`.


As an example, you can reproduce the experiment for Movies rationales with 1000
explanations on the 11B model by running (from this directory):

```
export PROJECT=yourproject
export ZONE=yourzone
export BUCKET=yourbucket
export TPU=yourtpu

ctpu up   --name=$TPU   --project=$PROJECT  --zone=$ZONE   --tpu-size=v3-256   --tpu-only   --noconf

TASK=movie_rationales_take1000_explanations
PRETRAINED_DIR=gs://t5-data/pretrained_models/11B
PRETRAINED_STEPS=1000000
FINETUNE_STEPS=20000
MODEL_DIR="${BUCKET}/${TASK}/xxl_ssm"

# Run fine-tuning
t5_mesh_transformer \
  --tpu="${TPU}" \
  --gcp_project="${PROJECT}" \
  --tpu_zone="${ZONE}" \
  --model_dir="${BUCKET}/movies_1000/11B/" \
  --gin_file="dataset.gin" \
  --gin_file="${PRETRAINED_DIR}/operative_config.gin" \
  --gin_file="wt5/gin/sequence_lengths/movie_rationales_v010.gin" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = '8x16' \
  --gin_param="MIXTURE_NAME = '${TASK}' \
  --gin_param="mesh_train_dataset_fn.use_cached=False" \
  --gin_param="utils.run.save_checkpoints_steps=100" \
  --gin_param="utils.run.batch_size=('tokens_per_batch', 65536)" \
  --gin_param="utils.run.train_steps=$((PRETRAINED_STEPS+FINETUNE_STEPS))" \
  --gin_param="utils.run.init_checkpoint='${PRETRAINED_DIR}/model.ckpt-${PRETRAINED_STEPS}' \
  --gin_param="utils.run.learning_rate_schedule=@learning_rate_schedules.constant_learning_rate" \
  --gin_param="constant_learning_rate.learning_rate=1e-3" \
  --t5_tfds_data_dir="${BUCKET}/t5-tfds" \
  --module_import="wt5.tasks" \
  --module_import="wt5.mixtures"

# Run eval
TASK=movie_rationales_eval_v010
t5_mesh_transformer \
  --tpu="${TPU}" \
  --gcp_project="${PROJECT}" \
  --tpu_zone="${ZONE}" \
  --model_dir="${BUCKET}/movies_1000/11B/" \
  --gin_file="dataset.gin" \
  --gin_file="${BUCKET}/movies_1000/11B/operative_config.gin" \
  --gin_file="wt5/gin/sequence_lengths/movie_rationales_v010.gin" \
  --gin_file="eval.gin" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = '8x16' \
  --gin_param="MIXTURE_NAME = '${TASK}' \
  --gin_param="mesh_eval_dataset_fn.use_cached=False" \
  --gin_param="utils.run.dataset_split = 'validation' \
  --gin_param="utils.run.batch_size=('tokens_per_batch', 65536)" \
  --gin_param="utils.run.eval_checkpoint_step='all' \
  --t5_tfds_data_dir="${BUCKET}/t5-tfds" \
  --module_import="wt5.tasks" \
  --module_import="wt5.mixtures"
```

The remaining experiments are shown in the [tasks.py](wt5/tasks.py) and [mixtures.py](wt5/mixtures.py) files.

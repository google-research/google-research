# Primer: Searching for Efficient Transformers for Language Modeling

This repository contains the open sourced code for the T5 experiments in
"Primer: Searching for Efficient Transformers for Language Modeling."

## Launching Experiments

This code is built on top of the [T5 library](https://pypi.org/project/t5/).
Specifically, we leverage the `t5_mesh_transformer` training program compatible
with [Google Cloud](https://cloud.google.com/sdk/docs). See the
[T5 GitHub repository](https://github.com/google-research/text-to-text-transfer-transformer)
for more information on how to configure Google Cloud resources to use the
`t5_mesh_transformer` program.

Note, this is built on the latest version of MeshTF:
```
pip install -e "git+https://github.com/tensorflow/mesh.git#egg=mesh-tensorflow"
```

Here we provide an example command for training Primer on C4. First, create
a Google Cloud TPU to use:
```
export PROJECT=<project>
export ZONE=<zone>
export BUCKET=<bucket>
export TPU_NAME=<tpu>

ctpu up   --name=$TPU_NAME   --project=$PROJECT  --zone=$ZONE   --tpu-size=v3-8 \
   --tf-version=2.6.0   --noconf
```

After SSH-ing into the corresponding VM, you can launch training. Note,
our dependencies are added via the flag
`--module_import="t5_imports"`:
```
export TASK=c4_v220_autoregressive_language_modeling
export MODEL_NAME="medium_primer"
export MODEL_DIR="${BUCKET}/${MODEL_NAME}_${TASK}"
export DATA_DIR="${BUCKET}/data"

# Run training job.
t5_mesh_transformer   --tpu="${TPU_NAME}"   --gcp_project="${PROJECT}" \
  --tpu_zone="${ZONE}"   --model_dir="${MODEL_DIR}"  --gin_file="dataset.gin" \
  --t5_tfds_data_dir="${DATA_DIR}" \
  --gin_file="models/defaults.gin"   --gin_file="models/${MODEL_NAME}.gin" \
  --gin_param="utils.tpu_mesh_shape.model_parallelism = 1" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = \"2x2\"" \
  --gin_param="MIXTURE_NAME = \"${TASK}\"" \
  --gin_param="utils.run.train_steps = 1000000" \
  --module_import="t5_imports" \
  --gin_location_prefix="gin/" \
  --gin_param="utils.run.batch_size = (\"tokens_per_batch\", 65536)" \
  --gin_param="run.sequence_length = {'inputs': 1, 'targets': 512}" \
  --gin_file="learning_rate_schedules/rsqrt_no_ramp_down.gin" \
  --gin_file="objectives/lm.gin" \
  --gin_param="run.model_type = 'lm'" \
  --gin_param="utils.run.save_checkpoints_steps = 2400" \
  --gin_param="utils.serialize_num_microbatches.tokens_per_microbatch_per_replica = 8192"

# Run evaluation job.
t5_mesh_transformer   --tpu="${TPU_NAME}"   --gcp_project="${PROJECT}" \
  --tpu_zone="${ZONE}"   --model_dir="${MODEL_DIR}"  --gin_file="dataset.gin" \
  --t5_tfds_data_dir="${DATA_DIR}" \
  --gin_file="models/defaults.gin"   --gin_file="models/${MODEL_NAME}.gin" \
  --gin_param="utils.tpu_mesh_shape.model_parallelism = 1" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = \"2x2\"" \
  --gin_param="MIXTURE_NAME = \"${TASK}\"" \
  --module_import="t5_imports" \
  --gin_location_prefix="gin/" \
  --gin_param="run.sequence_length = {'inputs': 1, 'targets': 512}" \
  --gin_param="run.model_type = 'lm'" \
  --gin_file="perplexity_eval.gin" \
  --gin_param="utils.run.save_checkpoints_steps = 2400" \
  --gin_param="utils.serialize_num_microbatches.tokens_per_microbatch_per_replica = 8192" \
  --gin_param="utils.run.mode = \"perplexity_eval\"" \
  --gin_param="utils.run.dataset_split = \"validation\"" \
  --gin_param="utils.run.eval_checkpoint_step = \"all\""
```

## Experiment Configuration Details

To match the experiments in the Primer paper, the following model and
dataset pairings should be used:

  - LM1B: Any of the "small" models.
  - C4: Any of the "medium" or "large" models; Switch Transformer and Switch
        Primer; and either of the Synthesizer models.
  - PG19: Any of the "medium" sized models.

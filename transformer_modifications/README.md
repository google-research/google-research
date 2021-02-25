# Do Transformer Modifications Transfer Across Implementations and Applications?

This repository contains the code for reproducing the experiments in
[Do Transformer Modifications Transfer Across Implementations and Applications?](https://arxiv.org/abs/2102.11972)

## Table of Contents

* [Usage](#usage)
* [How to cite](#how-to-cite)

## Usage

To run this code, you need to install the
[t5 library](https://pypi.org/project/t5/). General instructions for training, fine-tuning, evaluation, and exporting models for inference can be found in the [t5 repo](https://github.com/google-research/text-to-text-transfer-transformer). In order to use the additional tasks and mixtures provided in this library with the `t5_mesh_transformer` commands, run from this directory and add the flag `--module_import="transformer_modifications.mixtures"`.

As an example, you can reproduce the pre-training experiment for the SwiGLU
activation function (with the relative attention version of vanilla transformer)
by running:

```
export PROJECT=yourproject
export ZONE=yourzone
export BUCKET=yourbucket
export TPU=yourtpu

ctpu up   --name=$TPU   --project=$PROJECT  --zone=$ZONE   --tpu-size=v3-8 \
   --tf-version=2.3 --tpu-only   --noconf

TASK=c4_v231_unsupervised_en32k
BASE_MODEL_NAME="vanilla_transformer_rel_bias_shared"
MODEL_NAME="swiglu"
MODEL_DIR="${BUCKET}${MODEL_NAME}"

# Run pre-training
t5_mesh_transformer \
  --tpu="${TPU}" \
  --gcp_project="${PROJECT}" \
  --tpu_zone="${ZONE}" \
  --model_dir="${MODEL_DIR} "\
  --gin_file="dataset.gin" \
  --gin_file="models/${BASE_MODEL_NAME}.gin" \
  --gin_file="models/${MODEL_NAME}.gin" \
  --gin_param="utils.tpu_mesh_shape.model_parallelism = 1" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = \"2x2\"" \
  --gin_param="MIXTURE_NAME = \"${TASK}\"" \
  --gin_param="utils.run.train_steps = 524288" \
  --module_import="transformer_modifications.mixtures" \
  --gin_location_prefix="transformer_modifications/transformer_modifications/gin/" \
  --gin_param="utils.run.batch_size = (\"tokens_per_batch\", 65536)"  \
  --gin_file="defaults.gin" \
  --gin_file="learning_rate_schedules/rsqrt_no_ramp_down.gin" \
  --gin_file="objectives/span_3_15_u_u.gin" \
  --gin_param="utils.run.save_checkpoints_steps = 2400" \
  --gin_param="utils.serialize_num_microbatches.tokens_per_microbatch_per_replica = 8192"

# Run perplexity eval
t5_mesh_transformer \
  --tpu="${TPU}" \
  --gcp_project="${PROJECT}" \
  --tpu_zone="${ZONE}" \
  --model_dir="${MODEL_DIR} "\
  --gin_file="dataset.gin" \
  --gin_file="models/vanilla_transformer_rel_bias_shared.gin" \
  --gin_file="models/swiglu.gin" \
  --gin_param="utils.tpu_mesh_shape.model_parallelism = 1" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = \"2x2\"" \
  --gin_param="MIXTURE_NAME = \"c4_v231_unsupervised_en32k\"" \
  --module_import="transformer_modifications.mixtures" \
  --gin_location_prefix="transformer_modifications/transformer_modifications/gin/" \ 
  --gin_file="defaults.gin" \
  --gin_file="perplexity_eval.gin" \
  --gin_param="utils.run.save_checkpoints_steps = 2400" \
  --gin_param="utils.serialize_num_microbatches.tokens_per_microbatch_per_replica = 8192" \
  --gin_param="utils.run.mode = \"perplexity_eval\"" \
  --gin_param="utils.run.dataset_split = \"validation\"" \
  --gin_param="utils.run.eval_checkpoint_step = \"all\""
```

To fine-tune the above pre-trained model on the Extreme Summarization dataset (XSum), you can run the following command:

```
export PYTHONPATH="/shared/at5/:/shared/google-research/"
t5_mesh_transformer \
  --tpu="${TPU}" \
  --gcp_project="${PROJECT}" \
  --tpu_zone="${ZONE}" \
  --model_dir="${MODEL_DIR}/xsum_v110" \
  --gin_file="dataset.gin" \
  --gin_file="models/${BASE_MODEL_NAME}.gin" \
  --gin_file="models/${MODEL_NAME}.gin" \
  --gin_param="utils.tpu_mesh_shape.model_parallelism = 1" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = \"2x2\"" \
  --gin_param="MIXTURE_NAME = \"xsum_v110\"" \
  --gin_param="utils.run.train_steps = 786432" \
  --module_import="transformer_modifications.mixtures" \
  --gin_location_prefix="transformer_modifications/transformer_modifications/gin/" \ 
  --gin_param="utils.run.batch_size = (\"tokens_per_batch\", 65536)"  \
  --gin_file="${MODEL_DIR}/operative_config.gin" \
  --gin_file="sequence_lengths/xsum_v110.gin" \
  --gin_param="utils.run.init_checkpoint = \"${MODEL_DIR}/model.ckpt-524288\"" \
  --gin_param="utils.run.save_checkpoints_steps = 2400" \
  --gin_param="utils.serialize_num_microbatches.tokens_per_microbatch_per_replica = 8192" \
  --gin_param="dropout_rate = 0.1" \
  --gin_param="utils.run.learning_rate_schedule=[@learning_rate_schedules.constant_learning_rate,@learning_rate_schedules.linear_warmup]" \
  --gin_param="constant_learning_rate.learning_rate=0.0005" \
  --gin_param="learning_rate_schedules.linear_warmup.steps_or_fraction=564288"
```

To run supervised training experiments on WMT with SwiGLU activation function,
you can use the following command:

```
export PYTHONPATH="/shared/at5/:/shared/google-research/"
t5_mesh_transformer \
  --tpu="${TPU}" \
  --gcp_project="${PROJECT}" \
  --tpu_zone="${ZONE}" \
  --model_dir="${MODEL_DIR}_wmt"\
  --gin_file="dataset.gin" \
  --gin_file="models/${BASE_MODEL_NAME}.gin" \
  --gin_file="models/${MODEL_NAME}.gin" \
  --gin_param="utils.tpu_mesh_shape.model_parallelism = 1" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = \"2x2\"" \
  --gin_param="MIXTURE_NAME = \"wmt_t2t_ende_v003_vocab_37000\"" \
  --gin_param="utils.run.train_steps = 150000" \
  --module_import="transformer_modifications.mixtures" \
  --gin_location_prefix="transformer_modifications/transformer_modifications/gin/" \ 
  --gin_param="utils.run.batch_size = (\"tokens_per_batch\", 65536)"  \
  --gin_file="${MODEL_DIR}/operative_config.gin" \
  --gin_param="utils.run.save_checkpoints_steps = 2400" \
  --gin_param="utils.serialize_num_microbatches.tokens_per_microbatch_per_replica = 8192" \
  --gin_param="dropout_rate = 0.1" \
  --gin_param="utils.run.learning_rate_schedule=[@learning_rate_schedules.constant_learning_rate,@learning_rate_schedules.linear_warmup]" \
  --gin_param="constant_learning_rate.learning_rate=0.0005" \
```
# How to Cite

If you extend or use this work, please cite the [paper](https://arxiv.org/abs/2004.14546) where it was introduced:

```
@misc{narang2021transformer,
      title={Do Transformer Modifications Transfer Across Implementations and Applications?},
      author={Sharan Narang and Hyung Won Chung and Yi Tay and William Fedus and Thibault Fevry and Michael Matena and Karishma Malkan and Noah Fiedel and Noam Shazeer and Zhenzhong Lan and Yanqi Zhou and Wei Li and Nan Ding and Jake Marcus and Adam Roberts and Colin Raffel},
      year={2021},
      eprint={2102.11972},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

# Compositional Generalization in Semantic Parsing

Code and details for reproducing results for the paper ["Compositional
Generalization in Semantic Parsing: Pre-training vs. Specialized Architectures"]
(https://arxiv.org/abs/2007.08970)

## T5 fine-tuning instructions

Below are instructions for fine-tuning a T5-small model on the MCD1 split of
CFQ. These instructions can easily be modified to fine-tune T5 on any split of
both the SCAN and CFQ dataset.

### Before you begin

Before starting this tutorial, check that your Google Cloud project is correctly
set up. For more information, see [Set up an account and a Cloud TPU project](https://cloud.google.com/tpu/docs/setup-gcp-account).

We recommend using CTPU to create a VM and a TPU device. See [the quickstart](https://cloud.google.com/tpu/docs/quickstart) for more details.

The [T5 Github page](https://github.com/google-research/text-to-text-transfer-transformer) contains further instructions for setting up the TPU device using CTPU.

### Installing dependencies

Once you are logged into your Google Cloud VM, install T5 for GCP.

```
pip3 install t5[gcp]
```

Then download the custom T5 tasks from this repository in order to run
fine-tuning on SCAN and CFQ, as well as the preprocessing library that is part
of CFQ.

```
GR_REPO="https://raw.githubusercontent.com/google-research/google-research"
wget "${GR_REPO}/master/cfq/preprocess.py" \
"${GR_REPO}/master/cfq_pt_vs_sa/cfq_scan_tasks.py"
```

Next, make sure these two files can be used by the T5 binary by putting them 
into a directory and adding it to `PYTHONPATH`.

```
mkdir t5_utils
mv preprocess.py cfq_scan_tasks.py t5_utils
export PYTHONPATH=$PYTHONPATH:$(pwd)/t5_utils
```

### Fine-tuning.

Make sure the following environment variables are set:

*   `TPU_NAME`: The name of your TPU device.

*   `PROJECT`: The name of your Google Cloud project.

*   `ZONE`: Your project zone.

*   `MODEL_DIR`: Location where the model will be saved (in a Cloud bucket).

*   `DATA_DIR`: Location where the dataset data will be saved.

*   `TPU_SIZE`: Size of your TPU (e.g., 2x2).

We used the following settings for model parallelism and TPU topologies when
doing our experiments.

| T5 size | model parallelism | TPU topology |
|---------|-------------------|--------------|
| small   | 1                 | 4x4          |
| base    | 2                 | 8x4          |
| large   | 8                 | 8x16         |
| 3B      | 8                 | 8x16         |
| 11B     | 8                 | 8x16         |

Then run the following command to start fine-tuning:

```
DATASET=cfq
SPLIT=mcd1
T5_SIZE=small
t5_mesh_transformer  \
  --tpu="${TPU_NAME}" \
  --gcp_project="${PROJECT}" \
  --tpu_zone="${ZONE}" \
  --model_dir="${MODEL_DIR}" \
  --t5_tfds_data_dir="${DATA_DIR}" \
  --gin_file="dataset.gin" \
  --gin_file="models/bi_v1.gin" \
  --gin_param="utils.tpu_mesh_shape.model_parallelism = 1" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = '${TPU_SIZE}'" \
  --gin_file="gs://t5-data/pretrained_models/${T5_SIZE}/operative_config.gin" \
  --gin_param="MIXTURE_NAME = '${DATASET}_${SPLIT}'" \
  --module_import=t5_utils.cfq_scan_tasks
```

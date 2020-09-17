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

**NOTE** Please make sure your VM has at least 30Gb of RAM, otherwise
preprocessing the CFQ dataset will fail.

The [T5 Github page](https://github.com/google-research/text-to-text-transfer-transformer) contains further instructions for setting up the TPU device using CTPU.

### Installing dependencies

Once you are logged into your Google Cloud VM, install T5 for GCP.

```
pip3 install t5[gcp]
```

Then download the custom T5 tasks from this repository in order to run
fine-tuning on SCAN and CFQ, and make sure the tasks can be used by the T5 binary by
putting them into a directory and adding it to `PYTHONPATH`.

```
GR_REPO="https://raw.githubusercontent.com/google-research/google-research"
wget "${GR_REPO}/master/cfq_pt_vs_sa/cfq_scan_tasks.py"
mkdir t5_cfq_scan
mv cfq_scan_tasks.py t5_cfq_scan
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

### Fine-tuning.

The preprocessing in `cfq_scan_tasks.py` uses `tf.py_function`, which doesn't
work on headless TPU devices. Therefore we cache the datasets first using T5's [cache_tasks_main.py](https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/cache_tasks_main.py).

First download the dataset and split to your local disc using TFDS.

```
python3 -c "import tensorflow_datasets as tfds; tfds.load('cfq/mcd1')"
```

The dataset will be cached to `~/tensorflow_datasets` by default. Next we
cache the dataset.

```
TFDS_LOCAL_PATH=$USER/tensorflow_datasets
# Change this to a directory in a Cloud bucket.
CACHE_DIR=gs://<YOUR_CLOUD_BUCKET>/cache

# Clone the repository locally.
git clone https://github.com/google-research/text-to-text-transfer-transformer.git

python3 text-to-text-transfer-transformer/t5/data/cache_tasks_main.py \
  --tasks=cfq_mcd1 \
  --output_cache_dir=$CACHE_DIR \
  --module_import=t5_cfq_scan.cfq_scan_tasks \
  --tasks_additional_cache_dirs=$TFDS_LOCAL_PATH
```

Make sure the following environment variables are set:

*   `TPU_NAME`: The name of your TPU device.

*   `PROJECT`: The name of your Google Cloud project.

*   `ZONE`: Your project zone.

*   `MODEL_DIR`: Location where the model will be saved (in a Cloud bucket).

*   `DATA_DIR`: Location where the dataset data will be saved.

*   `TPU_SIZE`: Size of your TPU (e.g., 2x2).

*   `CACHE_DIR`: Google Cloud directory where the dataset is cached.

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
  --gin_param="utils.tpu_mesh_shape.model_parallelism = 1" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = '${TPU_SIZE}'" \
  --gin_file="gs://t5-data/pretrained_models/${T5_SIZE}/operative_config.gin" \
  --gin_param="MIXTURE_NAME = '${DATASET}_${SPLIT}'" \
  --module_import=t5_cfq_scan.cfq_scan_tasks \
  --additional_task_cache_dirs="${CACHE_DIR}" \
  --gin_param="mesh_train_dataset_fn.use_cached = True"
```

### Evaluation

Assuming the model-finetuned for 262,144 steps (the default), and all bash flags
mentioned above are set, evaluation can be run with the following command:

```
t5_mesh_transformer \
  --tpu="${TPU_NAME}" \
  --gcp_project="${PROJECT}" \
  --tpu_zone="${ZONE}" \
  --model_dir="${MODEL_DIR}" \
  --gin_file="${MODEL_DIR}/operative_config.gin" \
  --t5_tfds_data_dir=${DATA_DIR} \
  --gin_file="eval.gin" \
  --gin_file="beam_search.gin" \
  --gin_param="run.dataset_split = 'validation'" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = '${TPU_SIZE}'" \
  --gin_param="eval_checkpoint_step = 262144" \
  --gin_param="MIXTURE_NAME = 'cfq_${SPLIT}'" \
  --module_import=t5_cfq_scan.cfq_scan_tasks
```

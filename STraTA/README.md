# STraTA: Self-Training with Task Augmentation

This repository contains data and code for our [EMNLP 2021](https://2021.emnlp.org/) paper: [STraTA: Self-Training with Task Augmentation for Better Few-shot Learning](https://arxiv.org/abs/2109.06270). Our new implementation of STraTA typically yields better results than what reported in our paper.

**Note**: Our code can be used as a tool for automatic data labeling.

## Table of Contents

   * [Installation](#installation)
   * [Self-training](#self-training)
      * [Running self-training with a base model](#running-self-training-with-a-base-model)
      * [Hyperparameters for self-training](#hyperparameters-for-self-training)
      * [Distributed training](#distributed-training)
      * [Practical recommendations](#practical-recommendations)
   * [Task augmentation](#task-augmentation)
      * [T5 NLI data generation model checkpoints](#t5-nli-data-generation-model-checkpoints)
      * [Generating synthetic NLI data](#generating-synthetic-nli-data)
      * [Practical recommendations](#practical-recommendations)
   * [Comparison to our work](#comparison-to-our-work)
   * [Demo](#demo)
   * [FAQ](#faq)
   * [How to cite](#how-to-cite)

## Installation
This repository is tested on Python 3.8+, PyTorch 1.10+, and the 🤗 Transformers 4.16+.

You should install all necessary Python packages in a [virtual environment](https://docs.python.org/3/library/venv.html). If you are unfamiliar with Python virtual environments, please check out the [user guide](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).

Below, we create a virtual environment with the [Anaconda Python distribution](https://www.anaconda.com/products/distribution) and activate it.
```sh
conda create -n strata python=3.9
conda activate strata
```
Next, you need to install 🤗 Transformers. Please refer to [🤗 Transformers installation page](https://github.com/huggingface/transformers#installation) for a detailed guide.
```sh
pip install transformers
```
Finally, install all necessary Python packages for our self-training algorithm.

```sh
pip install -r requirements.txt
```
This will install PyTorch as a backend.

## Self-training
### Running self-training with a base model
The following example code shows how to run our self-training algorithm with a base model (e.g., `BERT`, `BERT` fine-tuned on `MNLI`, `BERT` produced by task augmentation) on the `SciTail` science entailment dataset, which has two classes `['entails', 'neutral']`. We assume that you have a data directory that includes some training data (e.g., `train.csv`), evaluation data (e.g., `eval.csv`), and unlabeled data (e.g., `infer.csv`).

```python
import os
from selftraining import selftrain

data_dir = '/path/to/your/data/dir'
parameters_dict = {
    'max_selftrain_iterations': 100,
    'model_name_or_path': '/path/to/your/base/model',  # could be the id of a model hosted by 🤗 Transformers
    'output_dir': '/path/to/your/output/dir',
    'train_file': os.path.join(data_dir, 'train.csv'),
    'infer_file': os.path.join(data_dir, 'infer.csv'),
    'eval_file': os.path.join(data_dir, 'eval.csv'),
    'evaluation_strategy': 'steps',
    'task_name': 'scitail',
    'label_list': ['entails', 'neutral'],
    'per_device_train_batch_size': 32,
    'per_device_eval_batch_size': 8,
    'max_length': 128,
    'learning_rate': 2e-5,
    'max_steps': 100000,
    'eval_steps': 1,
    'early_stopping_patience': 50,
    'overwrite_output_dir': True,
    'do_filter_by_confidence': False,
    # 'confidence_threshold': 0.3,
    'do_filter_by_val_performance': True,
    'finetune_on_labeled_data': False,
    'seed': 42,
}
selftrain(**parameters_dict)
```

**Note**: We checkpoint periodically during self-training. In case of preemptions, just re-run the above script and self-training will resume from the latest iteration.

### Hyperparameters for self-training
If you have development data, you might want to tune some hyperparameters for self-training.
Below are hyperparameters that could provide additional gains for your task.

  - `finetune_on_labeled_data`: If set to `True`, the resulting model from each self-training iteration is further fine-tuned on the original labeled data before the next self-training iteration. Intuitively, this would give the model a chance to "correct" ifself after being trained on pseudo-labeled data.
  - `do_filter_by_confidence`: If set to `True`, the pseudo-labeled data in each self-training iteration is filtered based on the model confidence. For instance, if `confidence_threshold` is set to `0.3`, pseudo-labeled examples with a confidence score less than or equal to `0.3` will be discarded. Note that `confidence_threshold` should be greater or equal to `1/num_labels`, where `num_labels` is the number of class labels. Filtering out the lowest-confidence pseudo-labeled examples could be helpful in some cases.
  - `do_filter_by_val_performance`: If set to `True`, the pseudo-labeled data in each self-training iteration is filtered based on the current validation performance. For instance, if your validation performance is 80% accuracy, you might want to get rid of 20% of the pseudo-labeled data with the lowest the confidence scores.

### Distributed training
We strongly recommend distributed training with multiple accelerators. To activate distributed training, please try one of the following methods:

1. Run `accelerate config` and answer to the questions asked. This will save a `default_config.yaml` file in your cache folder for 🤗 Accelerate. Now, you can run your script with the following command:

```sh
accelerate launch your_script.py --args_to_your_script
```

2. Run your script with the following command:

```sh
python -m torch.distributed.launch --nnodes="{$NUM_NODES}" --nproc_per_node="{$NUM_TRAINERS}" --your_script.py --args_to_your_script
```

3. Run your script with the following command:

```sh
torchrun --nnodes="{$NUM_NODES}" --nproc_per_node="{$NUM_TRAINERS}" --your_script.py --args_to_your_script
```

### Practical recommendations
We recommend starting with a pre-trained `BERT` model first to see how it performs on your task. Next, you might want to try self-training with a `BERT` model fine-tuned on `MNLI` (you could use our [pre-trained models](https://console.cloud.google.com/storage/browser/gresearch/strata/bert-mnli)), i.e., fine-tuning `BERT` on `MNLI` before self-training it on your task. If `MNLI` turns out to helpful for your task, you could possibly achieve better
performance by applying task augmentation to obtain a stronger base model for self-training.

## Task augmentation
### T5 NLI data generation model checkpoints

We release the following `T5` NLI data generation model checkpoints used in our paper:

* **[`T5`-3B-NLI-entailment](https://console.cloud.google.com/storage/browser/gresearch/strata/t5-3b-nli/entailment)** (3 billion parameters)
* **[`T5`-3B-NLI-neutral](https://console.cloud.google.com/storage/browser/gresearch/strata/t5-3b-nli/neutral)** (3 billion parameters)
* **[`T5`-3B-NLI-contradiction](https://console.cloud.google.com/storage/browser/gresearch/strata/t5-3b-nli/contradiction)** (3 billion parameters)
* **[`T5`-3B-NLI-entailment_reversed](https://console.cloud.google.com/storage/browser/gresearch/strata/t5-3b-nli/entailment_reversed)** (3 billion parameters)
* **[`T5`-3B-NLI-neutral_reversed](https://console.cloud.google.com/storage/browser/gresearch/strata/t5-3b-nli/neutral_reversed)** (3 billion parameters)
* **[`T5`-3B-NLI-contradiction_reversed](https://console.cloud.google.com/storage/browser/gresearch/strata/t5-3b-nli/contradiction_reversed)** (3 billion parameters)

Note that our models were trained using a maximum sequence length of 128 for both the input and target sequences.

To obtain these models, we fine-tune the original `T5-3B` model on `MNLI` in a _text-to-text_ format. Specifically, each `MNLI` training example `(sentA, sentB) → label` is cast as `label: sentA → sentB`. The "reversed" models (with the suffix "-reversed") were trained on reversed examples `label: sentB → sentA`. During inference, each model is fed a `label` and a `source_text` in the format `label: input_text` as input (e.g., `entailment: the facts are accessible to you`), and it generates some `target_text` as output (e.g., `you have access to the facts`).

Once inference is done, you need to create NLI examples as `(input_text, target_text) → label`, or `(target_text, input_text) → label` if you use a "reversed" model.

### Generating synthetic NLI data
Please follow the [`T5` installation instructions](https://github.com/google-research/text-to-text-transfer-transformer#installation) to install `T5` and set up accelerators on Google Cloud Platform. Then, take a look at the [`T5` decoding instructions](https://github.com/google-research/text-to-text-transfer-transformer#decode) to get an idea on how to produce predictions from one of our model checkpoints.

You need to prepare a text file `inputs.txt` with one example per line, in the format `label: input_text` (e.g., `contradiction: his acting was really awful`).
The following example command generates 3 output samples per input using top-k sampling with `k=5`:

```sh
t5_mesh_transformer \
  --tpu="${TPU_NAME}" \
  --gcp_project="${PROJECT}" \
  --tpu_zone="${ZONE}" \
  --model_dir="${MODEL_DIR}" \
  --gin_file="${MODEL_DIR}/operative_config.gin" \
  --gin_file="infer.gin" \
  --gin_file="sample_decode.gin" \
  --gin_param="input_filename = '/path/to/inputs.txt'"\
  --gin_param="output_filename = '/path/to/outputs.txt'"\
  --gin_param="utils.decode_from_file.repeats = 3" \    # number of output samples per input
  --gin_param="utils.run.sequence_length = {'inputs': 128, 'targets': 128}" \
  --gin_param="infer_checkpoint_step = '1065536'" \    # 1000000 pre-training steps + 65536 fine-tuning steps
  --gin_param="utils.run.batch_size = ('sequences_per_batch', 64)" \
  --gin_param="Bitransformer.decode.temperature = 1.0" \
  --gin_param="Unitransformer.sample_autoregressive.temperature = 1.0" \
  --gin_param="Unitransformer.sample_autoregressive.sampling_keep_top_k = 5" \    # top-k
  --gin_param="utils.tpu_mesh_shape.tpu_topology = '${TPU_SIZE}'" \
```

Assume that the input file `inputs.txt` has 10 examples, you should get an output file `outputs.txt` with 30 output samples, where the `3i-2, 3i-1, 3i`^th output samples correspond to the `i`^th input example (`i=1,2,...,10`).

### Practical recommendations
We recommend the following practices for task augmentation:

1. Overgeneration. In our experiments, we perform overgeneration to get a large amount of synthetic NLI training data. We generate 100 output samples per input with `top-k (k = 40)` sampling. This could be expensive when you have a large amount of unlabeled data though.
2. Filtering. This is an important step to improve the quality of synthetic NLI training data. We use a `BERT` model fine-tuned on `MNLI` in the original format as an NLI classifier to filter synthetic training examples (you could use our [pre-trained models](https://console.cloud.google.com/storage/browser/gresearch/strata/bert-mnli) or any reliable NLI models available on [🤗 Models](https://huggingface.co/models)). We only keep an example if the NLI classifier's predicted probability exceeds a certain threshold.
3. Combining synthetic and realistic data. In our experiments, we use a two-stage training procedure where the model is first trained on the synthetic NLI data before being fine-tuned on the realistic `MNLI` data.

## Comparison to our work
To facilitate your evaluation, we release the `BERT` model checkpoints produced by task augmentation (TA) across datasets used in our few-shot experiments. Note that these models were trained on synthetic NLI data created using unlabeled texts from a target dataset. To avoid differences in evaluation methodology (e.g., training/development data subsets, number of random restarts, etc.), which can have a high impact on model performance in a low-data setting, you might want to perform our self-training algorithm on top of these model checkpoints using your own evaluation setup (e.g., data splits).

| Dataset | `BERT`-Base + TA |  `BERT`-Large + TA |
| :-- | :-- | :-- |
| SNLI | **[`BERT`-Base-synNLI-from-SNLI](https://storage.googleapis.com/gresearch/strata/bert-base-synnli-from-snli.zip)** | **[`BERT`-Large-synNLI-from-SNLI](https://storage.googleapis.com/gresearch/strata/bert-large-synnli-from-snli.zip)**
| QQP | **[`BERT`-Base-synNLI-from-QQP](https://storage.googleapis.com/gresearch/strata/bert-base-synnli-from-qqp.zip)** | **[`BERT`-Large-synNLI-from-QQP](https://storage.googleapis.com/gresearch/strata/bert-large-synnli-from-qqp.zip)**
| QNLI | **[`BERT`-Base-synNLI-from-QNLI](https://storage.googleapis.com/gresearch/strata/bert-base-synnli-from-qnli.zip)** | **[`BERT`-Large-synNLI-from-QNLI](https://storage.googleapis.com/gresearch/strata/bert-large-synnli-from-qnli.zip)**
| SST-2 | **[`BERT`-Base-synNLI-from-SST-2](https://storage.googleapis.com/gresearch/strata/bert-base-synnli-from-sst-2.zip)** | **[`BERT`-Large-synNLI-from-SST-2](https://storage.googleapis.com/gresearch/strata/bert-large-synnli-from-sst-2.zip)**
| SciTail | **[`BERT`-Base-synNLI-from-SciTail](https://storage.googleapis.com/gresearch/strata/bert-base-synnli-from-scitail.zip)** | **[`BERT`-Large-synNLI-from-SciTail](https://storage.googleapis.com/gresearch/strata/bert-large-synnli-from-scitail.zip)**
| SST-5 | **[`BERT`-Base-synNLI-from-SST-5](https://storage.googleapis.com/gresearch/strata/bert-base-synnli-from-sst-5.zip)** | **[`BERT`-Large-synNLI-from-SST-5](https://storage.googleapis.com/gresearch/strata/bert-large-synnli-from-sst-5.zip)**
| STS-B | **[`BERT`-Base-synNLI-from-STS-B](https://storage.googleapis.com/gresearch/strata/bert-base-synnli-from-sts-b.zip)** | **[`BERT`-Large-synNLI-from-STS-B](https://storage.googleapis.com/gresearch/strata/bert-large-synnli-from-sts-b.zip)**
| SICK-E | **[`BERT`-Base-synNLI-from-SICK-E](https://storage.googleapis.com/gresearch/strata/bert-base-synnli-from-sick-e.zip)** | **[`BERT`-Large-synNLI-from-SICK-E](https://storage.googleapis.com/gresearch/strata/bert-large-synnli-from-sick-e.zip)**
| SICK-R | **[`BERT`-Base-synNLI-from-SICK-R](https://storage.googleapis.com/gresearch/strata/bert-base-synnli-from-sick-r.zip)** | **[`BERT`-Large-synNLI-from-SICK-R](https://storage.googleapis.com/gresearch/strata/bert-large-synnli-from-sick-r.zip)**
| CR | **[`BERT`-Base-synNLI-from-CR](https://storage.googleapis.com/gresearch/strata/bert-base-synnli-from-cr.zip)** | **[`BERT`-Large-synNLI-from-CR](https://storage.googleapis.com/gresearch/strata/bert-large-synnli-from-cr.zip)**
| MRPC | **[`BERT`-Base-synNLI-from-MRPC](https://storage.googleapis.com/gresearch/strata/bert-base-synnli-from-mrpc.zip)** | **[`BERT`-Large-synNLI-from-MRPC](https://storage.googleapis.com/gresearch/strata/bert-large-synnli-from-mrpc.zip)**
| RTE | **[`BERT`-Base-synNLI-from-RTE](https://storage.googleapis.com/gresearch/strata/bert-base-synnli-from-rte.zip)** | **[`BERT`-Large-synNLI-from-RTE](https://storage.googleapis.com/gresearch/strata/bert-large-synnli-from-rte.zip)**

## Demo
Please check out `run.sh` to see how to perform our self-training algorithm with a `BERT` Base model produced by task augmentation on the SciTail science entailment dataset using 8 labeled examples per class. Please turn off the debug mode by setting `DEBUG_MODE_ON=False`. You can configure your training environment by specifying `NUM_NODES` and `NUM_TRAINERS` (number of processes per node). To launch the script, simply run `source run.sh`. For your reference, below are the results we got with different development sets using distributed training on a single compute note with 4 NVIDIA GeForce GTX 1080 Ti GPUs.

| Development file | # Development examples |  Accuracy |
| :-: | :-: | :-: |
| eval_16.csv | 16 | 87.50
| eval_256.csv | 256 | **92.97**
| eval.csv | 1304 | 92.15

## FAQ
### What should I do if I do not have enough computational resources to run `T5` to produce synthetic data?
In this case, you could fine-tune a model on an intermediate task (e.g., `MNLI` or a closely related task to your task) before using it for self-training on your task. In our experiments, self-training on top of `BERT` fine-tuned on `MNLI` performs competitively with `STraTA` in many cases.

## How to cite
If you extend or use this work, please cite the [paper](https://arxiv.org/abs/2109.06270) where it was introduced:

```bibtex
@inproceedings{vu-etal-2021-strata,
    title = "{ST}ra{TA}: Self-Training with Task Augmentation for Better Few-shot Learning",
    author = "Vu, Tu  and
      Luong, Minh-Thang  and
      Le, Quoc  and
      Simon, Grady  and
      Iyyer, Mohit",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.462",
    doi = "10.18653/v1/2021.emnlp-main.462",
    pages = "5715--5731",
}
```

## Disclaimer
This is not an officially supported Google product.

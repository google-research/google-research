# RevThink

This repository includes the source code for our paper [Reverse Thinking Makes LLMs Stronger Reasoners](https://arxiv.org/abs/2411.19865).

# Installation

This repository has been tested with Python 3.10.14. We recommend using Conda for environment management. You can set up the Conda environment using the following commands:

```
conda create --name revthink python=3.10.14 -y
conda activate revthink
pip install -r requirements.txt
```


# Run Experiments
### Args usages
* `--task`: which dataset to use, can be: `SQA`, `CSQA`, `ARC`, `MATH`, `GSM8K`, `TabMWP`, `ANLI`, `Date`.
* `--n`: the proportion of training data to use, can be `1-100`. Default: `100`.
* `--model`: the student model to train, can be: `mistral-7b`, `gemma-7b`.

### Extract data
All data files can be download from `https://drive.google.com/file/d/1zoi4KNb9vzl8vd8Xh0dHfJ7DwSQDiiK7/view?usp=share_link`,
which includes a `data.zip` with two directories: `/training_data` and `/test_data`. Extract them using the following command:

```
unzip data.zip
```

We use the dataset Date as an example below.


### Run

**Step 1 (Optional):** Use the following script to augment the original dataset by generating forward reasoning, backward questions, and backward reasoning using the teacher model, Gemini-1.5-Pro:
```
python augment_data.py --task SQA
```
If you wish to generate the data by yourself, you will need to set up Vertex AI following this [tutorial](https://cloud.google.com/vertex-ai/generative-ai/docs/start/quickstarts/quickstart-multimodal), and fill in the `project_id` and `location` in `utils.py`.

Note: This step is optional as we provide the augmented datasets in the `/training_data` directory.

**Step 2:** Train the student model using the following script:
```
CUDA_VISIBLE_DEVICES=0 python train.py --task SQA --n 100 --model mistral-7b 
```

**Step 3:** Evaluate the student model with the following script:
```
CUDA_VISIBLE_DEVICES=0 python evaluate.py --task SQA --n 100 --model mistral-7b 
```

For out-of-domain (OOD) evaluation, you can use the following tasks as the `--task` argument: `BoolQ`, `OBQA`, `ESNLI`, or `GSM8K-Rev`.

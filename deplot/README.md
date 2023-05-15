# DePlot: Visual Language Reasoning on Charts and Plots

Code and checkpoints for training the visual language models introduced in
in the papers
 * DePlot: One-shot visual language reasoning by plot-to-table translation
 * MatCha: Enhancing Visual Language Pretraining with Math Reasoning and Chart Derendering.

## Installation

To install DePlot, it is necessary to clone the google-research repo:
```
git clone https://github.com/google-research/google-research.git
```
From the `google_research` folder, you may install the necessary requirements
in a conda environment by executing:
```
conda create -n deplot python=3.9
conda activate deplot
pip install -r deplot/requirements.txt -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

Our checkpoints are fully compatible with the
[codebase](https://github.com/google-research/pix2struct/) from the
[Pix2Struct paper](https://arxiv.org/abs/2210.03347). Therefore, all of the
tools and instructions described in the documentation apply here as well.

Since we use some gin configurations from Pix2Struct, the repository needs to
be cloned at a directory of choice, which can be exported in a `PIX2STRUCT`
environment variable.

```
git clone https://github.com/google-research/pix2struct.git $PIX2STRUCT
```

## Models

We provide pre-trained models and fine-tuned models.


| Task                | GCS Path (Base)                                    |
| --------------------| -------------------------------------------------- |
| Pre-trained         | gs://deplot/models/base/matcha/v1                  |
| Chart-to-table      | gs://deplot/models/base/deplot/v1                  |
| ChartQA             | gs://deplot/models/base/chartqa/v1                 |
| PlotQA V1           | gs://deplot/models/base/plotqa_v1/v1               |
| PlotQA V2           | gs://deplot/models/base/plotqa_v2/v1               |
| Chart2Text Statista | gs://deplot/models/base/chart2text_statista/v1     |
| Chart2Text Pew      | gs://deplot/models/base/chart2text_pew/v1          |


## Inference

The checkpoints are fully compatible with Pix2Struct.
For testing and demoing purposes, inference may be run on CPU.
In that case, please set the `export JAX_PLATFORMS=''` environment variable
to run on cpu.

### Web Demo

While running this command, the web demo can be accessed
at `localhost:8080` (or any port specified via the `port` flag), assuming you
are running the demo locally. You can then upload your custom image and optional
prompt. To use a Plot-To-Table DePlot/MatCha model, you need to specify the
query as: "Generate underlying data table of the figure below:".

```
python -m pix2struct.demo \
  --gin_search_paths="${PIX2STRUCT}/pix2struct/configs,${PIX2STRUCT}" \
  --gin_file=models/pix2struct.gin \
  --gin_file=runs/inference.gin \
  --gin_file=sizes/base.gin \
  --gin.MIXTURE_OR_TASK_NAME="'dummy_pix2struct'" \
  --gin.TASK_FEATURE_LENGTHS="{'inputs': 4096, 'targets': 512}" \
  --gin.BATCH_SIZE=1 \
  --gin.CHECKPOINT_PATH="'gs://deplot/models/base/deplot/v1'"
```


## Disclaimer

This is not an official Google product.

## Contact information

For help or issues, please submit a GitHub issue.

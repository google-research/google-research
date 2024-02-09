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

Thanks to the Hugging Face team, we also have DePlot [[doc]](https://huggingface.co/docs/transformers/main/en/model_doc/deplot) and MatCha [[doc]](https://huggingface.co/docs/transformers/main/en/model_doc/matcha) implementations in the HF Transfermers library.

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

The models are also available at Hugging Face:

| Task                | HF Path                                                  |
| --------------------| -------------------------------------------------------- |
| Pre-trained         | https://huggingface.co/google/matcha-base                |
| Chart-to-table      | https://huggingface.co/google/deplot                     |
| ChartQA             | https://huggingface.co/google/matcha-chartqa             |
| PlotQA V1           | https://huggingface.co/google/matcha-plotqa-v1           |
| PlotQA V2           | https://huggingface.co/google/matcha-plotqa-v2           |
| Chart2Text Statista | https://huggingface.co/google/matcha-chart2text-statista |
| Chart2Text Pew      | https://huggingface.co/google/matcha-chart2text-pew      |


## Finetuning

Continue pretraining/finetuning of MatCha and DePlot is supported through
Hugging Face Transformers. Please see
[here](https://huggingface.co/docs/transformers/main/model_doc/matcha#finetuning)
for more instructions.

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

We also provide a DePlot+LLM [demo](https://huggingface.co/spaces/fl399/deplot_plus_llm)
and a MatCha chart QA [demo](https://huggingface.co/spaces/fl399/matcha_chartqa),
both hosted on Hugging Face Spaces.

## <a name="how-to-cite-deplot"></a>How to cite DePlot and MatCha?

You can cite the [DePlot paper](https://arxiv.org/abs/2212.10505) and the
[MatCha paper](https://arxiv.org/abs/2212.09662) as follows:

```
@inproceedings{liu-2022-deplot,
  title={DePlot: One-shot visual language reasoning by plot-to-table translation},
  author={Fangyu Liu and Julian Martin Eisenschlos and Francesco Piccinno and Syrine Krichene and Chenxi Pang and Kenton Lee and Mandar Joshi and Wenhu Chen and Nigel Collier and Yasemin Altun},
  year={2023},
  booktitle={Findings of the 61st Annual Meeting of the Association for Computational Linguistics},
  url={https://arxiv.org/abs/2212.10505}
}

@inproceedings{liu-2022-matcha,
  title={MatCha: Enhancing Visual Language Pretraining with Math Reasoning and Chart Derendering},
  author={Fangyu Liu and Francesco Piccinno and Syrine Krichene and Chenxi Pang and Kenton Lee and Mandar Joshi and Yasemin Altun and Nigel Collier and Julian Martin Eisenschlos},
  year={2023},
  booktitle={Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics},
  url={https://arxiv.org/abs/2212.09662}
}
```

## Disclaimer

This is not an official Google product.

## Contact information

For help or issues, please submit a GitHub issue.

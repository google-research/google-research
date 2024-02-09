# Code for "High-throughput ML-guided design of diverse single-domain antibodies against SARS-CoV-2"

## System requirements
No particular requirements; we recommend downloading the files and then loading
the notebook in the free Google colab environment. Alternatively, users can
choose to install python 3.11+ and a notebook viewer of their choice on their
machine.

## Installation
Install via pip from github:

```bash
pip install git+https://github.com/google-research/google-research.git#subdirectory=covid_vhh_design
```

Alternatively, clone the repository and install the module from there.


```bash
git clone https://github.com/google-research/google-research.git
cd google-research/covid_vhh_design/
pip install .
```

Typical install time on a "normal" desktop computer: none required if using
Google colab.


## Content
* `colab/covid_paper_plots.ipynb`: Code used to create the figures of the paper.
* `colab/data.ipynb`: Describes the data.
* `colab/model.ipynb`: Shows how to score sequences with the model that we used
    to design the 3rd library.
* `covid.py`: COVID-related code.
* `helper.py`: Helper functions.
* `utils.py`: Data pre-processing functions.
* `plotting.py`: Helper functions for making plots.

## Disclaimer

This is not an officially supported Google product.

Contact christofa@google.com, lcolwell@google.com, or zmariet@google.com for
questions.

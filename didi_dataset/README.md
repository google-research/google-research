# The Didi dataset: Digital Ink Diagram data

This repository contains a [Colab](https://colab.research.google.com/) notebook
that demonstrates how to handle the Digital ink data from the Didi dataset.

The dataset contains digital ink drawings of diagrams with dynamic drawing
information. The dataset aims to foster research in interactive graphical
symbolic understanding. The dataset was obtained using a prompted data
collection effort.

[Download the Didi dataset](https://storage.cloud.google.com/digital_ink_diagram_data)

We provide the raw data in [NDJSON](http://ndjson.org/) format as well as the
prompts in [png](https://console.cloud.google.com/storage/browser/digital_ink_diagram_data/png),
[dot](https://console.cloud.google.com/storage/browser/digital_ink_diagram_data/dot), and
[xdot](https://console.cloud.google.com/storage/browser/digital_ink_diagram_data/xdot) format.

The dataset and details about its construction and use are described in this
ArXiV paper:
[The Didi dataset: Digital Ink Diagram data](https://arxiv.org/abs/2002.09303).

## Visualizing and converting the data.

We are providing a [colab notebook](didi_dataset.ipynb) that
demonstrates how to read and visualize the data. It also provides functions to
convert the data to TFRecord files for easy use in
tensorflow.

## Training and evaluating a model

First download the Didi dataset. For this you can either download the raw data
or use our [demo colab](didi_dataset.ipynb) to convert the data into TFRecord
files.

Our paper gives more information about a potential train/validation/test split
of the data.

## Licenses

The data is licensed by Google LLC under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license. The code is
released under an
[Apache 2](https://github.com/google-research/google-research/blob/master/LICENSE)
license.

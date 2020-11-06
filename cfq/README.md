# Compositional Freebase Questions (CFQ)

This repository contains a leaderboard and code for training and evaluating ML
architectures on the Compositional Freebase Questions (CFQ) dataset.

The dataset can be downloaded from the following URL:

[Download the CFQ dataset](https://storage.cloud.google.com/cfq_dataset/cfq1.1.tar.gz)

The dataset and details about its construction and use are described in this ICLR 2020 paper: [Measuring Compositional Generalization: A Comprehensive Method on Realistic Data](https://openreview.net/forum?id=SygcCnNKwr).

## Leaderboard

Architectures are ranked by accuracy on MCD-MEAN, which is the average over the
accuracy of the three MCD splits. We also report average accuracy results and
95% confidence intervals on the individual MCD splits.

If you submit papers on CFQ, please consider sending a pull request to merge
your results onto the leaderboard.

|                             | MCD-MEAN      | MCD1         | MCD2         | MCD3         |
|-----------------------------|---------------|--------------|--------------|--------------|
| **HPD [3]**                 | **69.0**      | **79.6**     | **59.6**     | **67.8**     |
| T5-11B-mod [2]              | 42.1 +/- 9.1  | 61.6 +/- 12.4| 31.3 +/ 12.8 | 33.3 +/- 2.3 |
| T5-11B [2]                  | 40.9 +/- 4.3  | 61.4 +/- 4.8 | 30.1 +/- 2.2 | 31.2 +/- 5.7 |
| Evolved Transformer [2]     | 20.8 +/- 0.7  | 42.4 +/- 1.0 | 9.3 +/- 0.8  | 10.8 +/- 0.2 |
| Universal Transformer [1]   | 18.9 +/- 1.4  | 37.4 +/- 2.2 | 8.1 +/- 1.6  | 11.3 +/- 0.3 |
| Transformer [1]             | 17.9 +/- 0.9  | 34.9 +/- 1.1 | 8.2 +/- 0.3  | 10.6 +/- 1.1 |
| LSTM+Attention [1]          | 14.9 +/- 1.1  | 28.9 +/- 1.8 | 5.0 +/- 0.8  | 10.8 +/- 0.6 |
| CGPS [2]                    | 7.1 +/- 1.8   | 13.2 +/- 3.9 | 1.6 +/- 0.8  | 6.6 +/- 0.6  |
| Neural Shuffle Exchange [2] | 2.8 +/- 0.3   | 5.1 +/- 0.4  | 0.9 +/- 0.1  | 2.3 +/- 0.3  |

[1] Keysers, Daniel, Nathanael Schärli, Nathan Scales, Hylke Buisman, Daniel
Furrer, Sergii Kashubin, Nikola Momchev et al. ["Measuring Compositional
Generalization: A Comprehensive Method on Realistic Data."](https://openreview.net/forum?id=SygcCnNKwr) In *ICLR2019*.

[2] Daniel Furrer, Marc van Zee, Nathan Scales, Nathanael Schärli.
["Compositional Generalization in Semantic Parsing: Pre-training vs. Specialized
Architectures"](https://arxiv.org/abs/2007.08970) In *arXiv e-prints, arXiv:2007.08970* 2020.

[3] Yinuo Guo, Zeqi Lin, Jian-Guang Lou, Dongmei Zhang. ["Hierarchical Poset Decoding for Compositional Generalization in Language"](https://arxiv.org/abs/2010.07792) In *NeurIPS2020*.


## Requirements

This library requires Python3 and the following Python3 libraries:

*   [absl-py](https://pypi.org/project/absl-py/)
*   [tensorflow](https://www.tensorflow.org/)
*   [tensor2tensor](https://github.com/tensorflow/tensor2tensor)
*   [tensorflow-datasets](https://www.tensorflow.org/datasets)

We recommend getting [pip3](https://pip.pypa.io/en/stable/) and then running the
following command, which will install all required libraries in one go:

```shell
sudo pip3 install -r requirements.txt
```

Note that Tensor2Tensor is no longer updated and is based on Tensorflow 1 which
is only available for Python <= 3.7.

## Training and evaluating a model

In order to train and evaluate a model, run the following:

```shell
bash run_experiment.sh
```

This will download and preprocessing the dataset, then train an LSTM model with
attention on the random split of the CFQ dataset, after which it will directly
be evaluated.

NOTE This may take quite long and consume a lot of memory. It is tested on a
machine with 6-core/12-hyperthread CPUs at 3.7Ghz, and 64Gb RAM, which took
about 20 hours. Also note that this will consume roughly 35Gb of RAM during
preprocessing. The run-time can be sped up significantly by running Tensorflow
with GPU support.

The expected accuracy using the default setting of the script is 97.4 +/- 0.3.

In order to run a different model or try a different split, simply modify the
parameters in `run_experiment.sh`. See that file for additional details.

For the expected accuracies of the other splits and architectures, please see
the paper (Table 4). In the paper we report the averages and confidence
intervals based on 5 runs. For the MCD splits, these numbers vary between MCD1,
MCD2, and MCD3, and the numbers reported in Table 4 are the averages over the 3
splits. Accuracies vary between 5% and 37% over splits and architectures:

|      | LSTM+attention | Transformer | Universal Transformer |
|-------|--------------|--------------|--------------|
| MCD1  | 28.9 +/- 1.8 | 34.9 +/- 1.1 | 37.4 +/- 2.2 |
| MCD2  |  5.0 +/- 0.8 |  8.2 +/- 0.3 |  8.1 +/- 1.6 |
| MCD3  | 10.8 +/- 0.6 | 10.6 +/- 1.1 | 11.3 +/- 0.3 |

## SCAN MCD splits
We also publish the SCAN MCD splits from our paper. In order to run over those
please download the dataset from the [original source](https://github.com/brendenlake/SCAN),
set `dataset_local_path` to point to the tasks.txt file and adjust `split_path`
to point to one of the mcd.json files from our [scan archive](https://storage.cloud.google.com/cfq_dataset/scan-splits.tar.gz).

## Tensorflow datasets

Our dataset and splits are also part of [TensorFlow Datasets](https://www.tensorflow.org/datasets)
(as of v2.1.0). Using the data is as simple as:

```python
import tensorflow_datasets as tfds
data = tfds.load('cfq/mcd1')
```

## License

CFQ is released under the [CC-BY license](https://creativecommons.org/licenses/by/4.0/).

## Dataset Metadata

The following table is necessary for this dataset to be indexed by search
engines such as <a href="https://g.co/datasetsearch">Google Dataset Search</a>.
<div itemscope itemtype="http://schema.org/Dataset">
<table>
  <tr>
    <th>property</th>
    <th>value</th>
  </tr>
  <tr>
    <td>name</td>
    <td><code itemprop="name">Compositional Freebase Questions</code></td>
  </tr>
  <tr>
    <td>alternateName</td>
    <td><code itemprop="alternateName">cfq</code></td>
  </tr>
  <tr>
    <td>url</td>
    <td><code itemprop="url">https://github.com/google-research/google-research/tree/master/cfq</code></td>
  </tr>
  <tr>
    <td>description</td>
    <td><code itemprop="description">The Compositional Freebase Questions (CFQ)
      is a dataset that is specifically designed to measure compositional
      generalization. CFQ is a simple yet realistic, large dataset of natural
      language questions and answers that also provides for each question a
      corresponding SPARQL query against the Freebase knowledge base. This means
      that CFQ can also be used for semantic parsing.</code></td>
  </tr>
  <tr>
    <td>provider</td>
    <td>
      <div itemscope itemtype="http://schema.org/Organization" itemprop="provider">
        <table>
          <tr>
            <th>property</th>
            <th>value</th>
          </tr>
          <tr>
            <td>name</td>
            <td><code itemprop="name">Google</code></td>
          </tr>
          <tr>
            <td>sameAs</td>
            <td><code itemprop="sameAs">https://en.wikipedia.org/wiki/Google</code></td>
          </tr>
        </table>
      </div>
    </td>
  </tr>
  <tr>
    <td>license</td>
    <td>
      <div itemscope itemtype="http://schema.org/CreativeWork" itemprop="license">
        <table>
          <tr>
            <th>property</th>
            <th>value</th>
          </tr>
          <tr>
            <td>name</td>
            <td><code itemprop="name">CC BY-SA 3.0</code></td>
          </tr>
          <tr>
            <td>url</td>
            <td><code itemprop="url">https://creativecommons.org/licenses/by-sa/3.0/</code></td>
          </tr>
        </table>
      </div>
    </td>
  </tr>
  <tr>
    <td>citation</td>
    <td><code itemprop="citation">Daniel Keysers et al. "Measuring Compositional Generalization: A Comprehensive Method on Realistic Data" (2020). https://openreview.net/pdf?id=SygcCnNKwr</code></td>
  </tr>
</table>
</div>

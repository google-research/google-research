# PRIME

## Table of contents
<a href='#Description'>Description</a><br>
<a href='#Dataset'>Dataset</a><br>
<a href='#Principles'>AI Principles</a><br>
<a href='#Acknowledgement'>Acknowledgements</a><br>
<a href='#Citation'>How to cite</a><br>
<a href='#Disclaimer'>Disclaimer</a><br>


<a id='Description'></a>
## Description

An **introductory tutorial** for the PRIME algorithm is available as a Colaboratory
notebook:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-research/google-research/blob/master/prime/prime_colab.ipynb)

<a id='Dataset'></a>
## Dataset

|                  | # of Infeasible | # of Feasible | Max Runtime (ms) | Min Runtime (ms) | Average Runtime (ms) |
|------------------|-----------------|---------------|------------------|------------------|----------------------|
| MobileNetEdgeTPU |          384355 |        115711 |         16352.26 |           252.22 |               529.13 |
| MobilenetV2      |          744718 |        255414 |          7398.13 |           191.35 |               375.05 |
| MobilenetV3      |          797460 |        202672 |          7001.46 |           405.19 |               993.75 |
| M4               |          791984 |        208148 |         35881.35 |           335.59 |               794.33 |
| M5               |          698618 |        301514 |         35363.55 |           202.55 |               440.52 |
| M6               |          756468 |        243664 |          4236.90 |           127.79 |               301.74 |
| UNet             |          449578 |         51128 |        124987.51 |           610.96 |              3681.75 |
| T-RNN Dec        |          405607 |         94459 |          4447.74 |           128.05 |               662.44 |
| T-RNN Enc        |          410933 |         88880 |          5112.82 |           127.97 |               731.20 |


<a id='Principles'></a>

## Principles
This project adheres to [Google's AI principles](PRINCIPLES.md). By
participating, using or contributing to this project you are expected to adhere
to these principles.

<a id='Acknowledgement'></a>

## Acknowledgements

For their invaluable feedback and suggestions, we extend our gratitude to:

* Learn to Design Accelerators Team at Google Research
* Google EdgeTPU
* Vizier Team at Google Research
* Christof Angermueller
* Sheng-Chun Kao
* Samira Khan
* Xinyang Geng

<a id='Citation'></a>

## How to cite

If you use this dataset, please cite:

```
@inproceedings{prime:iclr:2022,
  title={Data-Driven Offline Optimization For Architecting Hardware Accelerators},
  author={Kumar, Aviral and Yazdanbakhsh, Amir and Hashemi, Milad and Swersky, Kevin and Levine, Sergey},
  booktitle={International conference on learning representations},
  year={2022},
}
```

<a id='Disclaimer'></a>

## Disclaimer

This is not an official Google product.

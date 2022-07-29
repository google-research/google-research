# Menger-RL

## Table of contents
<a href='#ProductImpact'>Product Impact</a><br>
<a href='#Principles'>AI Principles</a><br>
<a href='#Acknowledgement'>Acknowledgements</a><br>
<a href='#Citation'>How to cite</a><br>
<a href='#Disclaimer'>Disclaimer</a><br>

<a id='ProductImpact'></a>

## Product Impact

Menger was the first distributed RL framework (when introduced in 2020) that
trains on distributed TPU pods, and scales to a few thousand workers for data
collection and inference on CPU. Menger provides an infrastructure for RL
applications that include complex environments, for which large-scale and
high-performing distributed data collection (actors) and use of pod-level TPUs
for training massive data samples is crucial (learner).

Menger was evaluated and adopted by the Circuit Training team as one of the
distributed RL infrastructure candidates to create floorplans for TPU-v5.

<a id='Principles'></a>

## Principles
This project adheres to [Google's AI principles](PRINCIPLES.md). By
participating, using or contributing to this project you are expected to adhere
to these principles.

<a id='Acknowledgement'></a>

## Acknowledgements

For their invaluable feedback and suggestions, we extend our gratitude to:

* Robert Ormandi
* Ebrahim Songhori
* Shen Wang
* Albin Cassirer
* James Laudon
* Joe Jiang
* Sat Chatterjee
* Piotr Stanczyk
* Sabela Ramos
* TF-Agents and Circuit Training Teams

<a id='Citation'></a>

## How to cite

If you use this work, please cite:

```
@inproceedings{menger:arxiv:2022,
  title={Menger: Large-Scale Distributed Reinforcement Learning with a Case Study of Macro Placement},
  author={Yazdanbakhsh, Amir and Chen, Junchao and Zheng, Yu},
  booktitle={Arxiv},
  year={2022},
}
```

<a id='Disclaimer'></a>

## Disclaimer

This is not an official Google product.

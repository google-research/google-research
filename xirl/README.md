# XIRL

- [Overview](#overview)
- [Setup](#setup)
- [Datasets](#datasets)
- [Experiments: Reproducing Paper Results](#experiments-reproducing-paper-results)
- [Extending XIRL](#extending-xirl)
- [Acknowledgments](#acknowledgments)

## Overview

Code release for our CoRL 2021 conference paper:

<table><tr><td>
    <strong>
        <a href="https://x-irl.github.io/">
            XIRL: Cross-embodiment Inverse Reinforcement Learning
        </a><br/>
    </strong>
    Kevin Zakka<sup>1,3</sup>, Andy Zeng<sup>1</sup>, Pete Florence<sup>1</sup>, Jonathan Tompson<sup>1</sup>, Jeannette Bohg<sup>2</sup>, and Debidatta Dwibedi<sup>1</sup><br/>
    Conference on Robot Learning (CoRL) 2021
</td></tr></table>

<sup>1</sup><em>Robotics at Google,</em>
<sup>2</sup><em>Stanford University,</em>
<sup>3</sup><em>UC Berkeley</em>

---

This repository contains code for both representation learning and downstream reinforcement learning.

Along with this codebase, we're releasing two additional libraries which you may find useful for building on top of our work:

* [x-magical](https://github.com/kevinzakka/x-magical): our Gym-like benchmark extension of MAGICAL geared towards cross-embodiment imitation.
* [torchkit](https://github.com/kevinzakka/torchkit): a lightweight library containing useful boilerplate utilities like logging and model checkpointing.

For the latest updates, see: [x-irl.github.io](https://x-irl.github.io)

## Setup

We use Python 3.8 and [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for development. We recommend the below steps to create an environment and install the dependencies:

```bash
# Create and activate environment.
conda create -n xirl python=3.8
conda activate xirl

# Install dependencies.
pip install -r requirements.txt
```

## Datasets

## Experiments: Reproducing Paper Results

## Extending XIRL

## Acknowledgments

## Citation

If you find this code useful, consider citing our work:

```bibtex
@article{zakka2021xirl,
    title = {XIRL: Cross-embodiment Inverse Reinforcement Learning},
    author = {Zakka, Kevin and Zeng, Andy and Florence, Pete and Tompson, Jonathan and Bohg, Jeannette and Dwibedi, Debidatta},
    journal = {Conference on Robot Learning (CoRL)},
    year = {2021}
}
```

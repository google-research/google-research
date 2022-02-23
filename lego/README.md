# LEGO:Latent Execution-Guided Reasoning for Multi-Hop Question Answering on Knowledge Graphs

## Installation

First clone the repository and the [smore repo](https://github.com/google-research/smore), and install the package dependency required in the `requirements.txt`. 

Then navigate to the root folder of the project and do 

    git submodule update --init
    pip install -e .

## Examples

Please see the example script of each methods under `lego/experiments/scripts` folder. 

## Citations

If you use this repo, please cite the following paper.

```
@inproceedings{ren2021lego,
  title={Lego: Latent execution-guided reasoning for multi-hop question answering on knowledge graphs},
  author={Ren, Hongyu and Dai, Hanjun and Dai, Bo and Chen, Xinyun and Yasunaga, Michihiro and Sun, Haitian and Schuurmans, Dale and Leskovec, Jure and Zhou, Denny},
  booktitle={International Conference on Machine Learning},
  pages={8959--8970},
  year={2021},
  organization={PMLR}
}
```

## License
LEGO is licensed under the Apache License, Version 2.0.

This is not an officially supported Google product.

Contact hyren@cs.stanford.edu and hadai@google.com for questions about the repo.

# TIDE: Textual Identity Detection for Evaluating and Augmenting Classification and Language Models
This repository contains the core library used in the paper
[TIDE: Textual Identity Detection for Evaluating and Augmenting Classification and Language Models](https://arxiv.org/abs/2309.04027).

We leverage the TIDAL dataset to develop an identity annotation and augmentation
tool that can be used to improve the availability of identity context and the
effectiveness of ML fairness techniques.

The TIDAL dataset can be found at https://github.com/google-research-datasets/TIDAL.

## Installation

Install via pip from github:

```bash
 pip install git+https://github.com/google-research/google-research.git#subdirectory=tide_nlp
```

 Alternatively, clone the repository and install the module from there.

```bash
git clone https://github.com/google-research/google-research.git
cd google-research/tide_nlp/
pip install .
```


Finally, download the default spaCy model.

```bash
python -m spacy download en_core_web_sm
```

## Citations
If you would like to cite the paper/code, please use the following BibTeX entry:

```
@misc{klu2023tide,
      title={TIDE: Textual Identity Detection for Evaluating and Augmenting Classification and Language Models},
      author={Emmanuel Klu and Sameer Sethi},
      year={2023},
      eprint={2309.04027},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## License
TIDE is licensed under the Apache License, Version 2.0.

This is not an officially supported Google product.
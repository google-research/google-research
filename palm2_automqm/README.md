# The Devil is in the Errors: Leveraging Large Language Models for Fine-grained Machine Translation Evaluation
This repository contains output data from the paper "[The Devil is in the Errors: Leveraging Large Language Models for Fine-grained Machine Translation Evaluation](https://arxiv.org/abs/2308.07286)."

## Overview

The data can be downloaded from [this link](https://storage.googleapis.com/gresearch/palm2_automqm/metric-scores.zip).

The zip file contains the metric predictions on the WMT'22 dataset for the en-de, zh-en, and en-ru language pairs and the metric predictions on the WMT'19 dataset for the en-gu, gu-en, en-kk, and kk-en language pairs.

The score files are formatted so that they can be used directly by the [MTME library](https://github.com/google-research/mt-metrics-eval).
Each `.seg.score` file is a tsv with two columns.
The first column is the name of a translation system and the second column is a predicted score.
The files are sorted such that the first instance of a system corresponds to segment ID 0, the second instance to segment ID 1, etc.

Each `.sys.score` file is a tsv with two columns.
The first column is the name of a translation system and the second column is that system's system-level score, equal to the average of the segment-level scores from the `.seg.score` files.

The `.mqm.jsonl` files contain one serialized JSON per row.
Each JSON contains the predicted error spans for a given translation plus the system ID, segment ID, source, reference, hypothesis, etc.

The names of the files indicate whether the metric uses a reference (`-refA`) or not (`-src`).
The model name is also encoded in the file name.

The zero-shot prompting models are:

      - PaLM (540b)
      - PaLM-2 (Bison)
      - PaLM-2 (Unicorn)
      - Flan-PaLM-2 (Unicorn)

The finetuned models are:

      - PaLM-2 (Bison) as a regression model
      - PaLM-2 (Unicorn) as a regression model
      - PaLM-2 (Bison) as a generative classification model
      - PaLM-2 (Unicorn) as a generative classification model

The AutoMQM models are:

      - PaLM-2 (Bison)
      - PaLM-2 (Unicorn)

## Citation
If you use the data from this work, please cite the following paper:
```
@misc{fernandes2023devil,
      title={{The Devil is in the Errors: Leveraging Large Language Models for Fine-grained Machine Translation Evaluation}}, 
      author={Patrick Fernandes and Daniel Deutsch and Mara Finkelstein and Parker Riley and André F. T. Martins and Graham Neubig and Ankush Garg and Jonathan H. Clark and Markus Freitag and Orhan Firat},
      year={2023},
      eprint={2308.07286},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2308.07286}
}
```
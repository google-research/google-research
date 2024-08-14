# Introducing the NewsPaLM MBR and QE Dataset: LLM-Generated High-Quality Parallel Data Outperforms Traditional Web-Crawled Data
This repository contains the NewsPaLM dataset, which accompanies the paper [Introducing the NewsPaLM MBR and QE Dataset: LLM-Generated High-Quality Parallel Data Outperforms Traditional Web-Crawled Data](https://arxiv.org/abs/2408.06537).

## Overview

The data can be downloaded from [this link](https://storage.googleapis.com/gresearch/newspalm_mbr_qe/newspalm.zip).

The zip file contains the NewsPaLM datasets. The names of the files indicate the decoding strategy (`mbr_decoded`, `qe_reranked`, or `greedy_decoded`), the language pair (`en_de` or `de_en`), and whether the dataset is `sentence_level` or `blob_level` (multi-sentence).

Each file is a tsv with two columns. The first column is the source, and the second column is the translation.

## Citation
If you use the data from this work, please cite the following paper:

```
@misc{finkelstein2024introducingnewspalmmbrqe,
      title={Introducing the NewsPaLM MBR and QE Dataset: LLM-Generated High-Quality Parallel Data Outperforms Traditional Web-Crawled Data},
      author={Mara Finkelstein and David Vilar and Markus Freitag},
      year={2024},
      eprint={2408.06537},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2408.06537},
}
```


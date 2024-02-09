# mFACE: Multilingual Summarization with Factual Consistency Evaluation

This repository includes the human judgements data used in the [mFACE paper](https://arxiv.org/abs/2212.10622) (Findings of ACL 2023).

## Data

The data is provided in [this spreadsheet](https://drive.google.com/file/d/1fsIK2pLMnzeIYlVH3j4APVBdxz-Zu6V0/view?usp=sharing), with the following columns: 

* id - row id
* doc_id - The document id in XLSum
* lang - The language of the example	
* split - The XLSum split this exmample was drawn from (train/dev/test)	
* input - The input document
* summary - The reference summary
* system - The summarization system that produced the summary (human=the reference summary)
* q1_[min/avg/max] - the min/avg/max score for the question "Is the summary comprehensible?"
* q2_[min/avg/max] - the min/avg/max score for the question "Is all the information in the summary fully attributable to the article?"
* q3_[min/avg/max] - the min/avg/max score for the question "Is the summary a good summary of the article?"
* rouge1 - The ROUGE1 score for the summary against the reference.
* rouge2 - The ROUGE2 score for the summary against the reference.
* rougeL - The ROUGEL score for the summary against the reference.
* e2e_nli_logprob - The log probability for entailment between the input and the summary (using the NLI model described in the paper).
* e2e_nli_score - The probability for entailment  between the input and the summary (using the NLI model described in the paper).

Following the XLSum dataset, the data is licensed with the [CC BY-NC-SA 4.0](https://huggingface.co/datasets/csebuetnlp/xlsum#licensing-information) license.

## How to cite

If you use any of the material here, please cite the following papers:

```bibtex
@inproceedings{aharoni2023mface,
    title = "Multilingual Summarization with Factual Consistency Evaluation",
    author = "Roee Aharoni and Shashi Narayan and Joshua Maynez and Jonathan Herzig and Elizabeth Clark and Mirella Lapata",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    year = "2023",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/2212.10622",
}

@inproceedings{hasan-etal-2021-xl,
    title = "{XL}-Sum: Large-Scale Multilingual Abstractive Summarization for 44 Languages",
    author = "Hasan, Tahmid  and
      Bhattacharjee, Abhik  and
      Islam, Md. Saiful  and
      Mubasshir, Kazi  and
      Li, Yuan-Fang  and
      Kang, Yong-Bin  and
      Rahman, M. Sohel  and
      Shahriyar, Rifat",
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-acl.413",
    pages = "4693--4703",
}
```
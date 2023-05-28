# TrueTeacher
Official repository for the paper - [TrueTeacher: Learning Factual Consistency Evaluation with Large Language Models](https://arxiv.org/pdf/2305.11171v1.pdf)

This is not an officially supported Google product.

## Data

### Data Description

Our dataset contains model-generated summaries of articles from the CNN/DailyMail which are annotated for factual consistency using FLAN-PaLM 540B.
We used 5 different summarization models to generate the summaries: `T5-11B`, `T5-3B`, `T5-large`, `T5-base` and `T5-small` which were fine tuned on the XSum dataset.

Additional details and data description can be found in the paper.

### Data location
Our dataset is placed in a public Google Cloud Storage Bucket and can be downloaded from
this [link](https://storage.googleapis.com/gresearch/true_teacher/true_teacher_data.zip).

### Data format
We have a separate file for each summarization model.

Each file contains json lines with the following keys:

- `"id"` - internal id, can be discarded in most cases.
- `"cnndm_id"` - the original id from the CNN/DailyMail dataset (we used only the "train" split), this need to be used in order to retrieve the article.
- `"summary"` - the model generated summary.
- `"label"` - a binary label, "1" indicating a factually consistent summary.

Here is an example of a single data item:

```json
{
  "id": "t5-11b_a_200960",
  "cnndm_id": "f72048a23154de8699c307e2f41157abbfcae261",
  "summary": "Children's brains are being damaged by prolonged internet access, a former children's television presenter has warned."
  "label": "1",
}
```

### Data License
Our dataset is licensed under the **Creative Commons Attribution-NonCommercial 4.0** International License.

To view a copy of this license, visit http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.


## Citing TrueTeacher
If you find this useful for your work, please use the following citation:

```
@article{gekhman2023trueteacher,
  title={Trueteacher: Learning factual consistency evaluation with large language models},
  author={Gekhman, Zorik and Herzig, Jonathan and Aharoni, Roee and Elkind, Chen and Szpektor, Idan},
  journal={arXiv preprint arXiv:2305.11171},
  year={2023}
}
```
# TrueTeacher
Official repository for the paper - [TrueTeacher: Learning Factual Consistency Evaluation with Large Language Models](https://arxiv.org/pdf/2305.11171v1.pdf)

This is not an officially supported Google product.

## Data

The TrueTeacher dataset contains model-generated summaries of articles from the train split of the **CNN/DailyMail** dataset [(Hermann et al., 2015)](https://proceedings.neurips.cc/paper_files/paper/2015/file/afdec7005cc9f14302cd0474fd0f3c96-Paper.pdf)
which are annotated for factual consistency using **FLAN-PaLM 540B** [(Chung et al.,2022)](https://arxiv.org/pdf/2210.11416.pdf).
Summaries were generated using summarization models with different capacities, which were created by fine-tuning **T5** [(Raffel et al., 2020)](https://jmlr.org/papers/volume21/20-074/20-074.pdf) on the **XSum** dataset [(Narayan et  al.,  2018)](https://aclanthology.org/D18-1206.pdf).
We used the following 5 capacities: T5-11B, T5-3B, T5-large, T5-base and T5-small.

The Dataset is available in 2 locations:

- In Hugging Face datasets ([datasets/google/trueteacher](https://huggingface.co/datasets/google/trueteacher)).
- In a public Google Cloud Storage Bucket. Can be downloaded from
this [link](https://storage.googleapis.com/gresearch/true_teacher/true_teacher_data.zip).

The Hugging Face variant should be slightly easier to use, we provide documentation and code snippets in [datasets/google/trueteacher](https://huggingface.co/datasets/google/trueteacher).

The variant in Google Cloud Storage Bucket is comprised of a separate file for each summarization model. Each file contains json lines with the following keys:

- `"cnndm_id"` - The original id from the CNN/DailyMail dataset, this need to be used in order to retrieve the corresponding article from CNN/DailyMail (which was used as the grounding document).
- `"summary"` - The model-generated summary.
- `"label"` - A binary label ('1' - Factualy Consistent, '0' - Factualy Inconsistent).

Here is an example of a single data item:

```json
{
  "cnndm_id": "f72048a23154de8699c307e2f41157abbfcae261",
  "summary": "Children's brains are being damaged by prolonged internet access, a former children's television presenter has warned.",
  "label": "1",
}
```

Additional details and data description can be found in the paper.

## Model

Our main model for factual consistency evaluation is available in Hugging Face ([google/t5_11b_trueteacher_and_anli](https://huggingface.co/google/t5_11b_trueteacher_and_anli)), with the corresponding documentation and code snippets.

Additional details and model description can be found in the paper.



## License
Our dataset and model are licensed under the **Creative Commons Attribution-NonCommercial 4.0** International License.

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
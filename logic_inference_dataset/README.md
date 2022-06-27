# LogicInference Dataset

This repository contains the Python code used to generate the `LogicInference`
dataset. `LogicInference` is a dataset designed to evaluate the ability of
models to perform logical inference. The dataset focuses on inference using
propositional logic and a small subset of first-order logic, represented both in
semi-formal logical notation, and in natural language. `LogicInference` has two
main long-term goals: (1) to evaluate the ability of models to perform logical
inference, and the degree to which inference chains are real or hallucinated,
and (2) to assess whether learning logical inference abilities in the abstract
(e.g., getting better in this dataset) would then transfer to other real-world
tasks.

For a detailed description of the dataset, please check the following paper:
https://openreview.net/pdf?id=HAGeIS_Lcg9 (arXiv preprint: https://arxiv.org/abs/2203.15099 )

Please cite as:

```
@inproceedings{ontanon2022logicinference,
  url = {https://openreview.net/pdf?id=HAGeIS_Lcg9},
  author = {Onta\~{n}\'{o}n, Santiago and Ainslie, Joshua and Cvicek, Vaclav and Fisher, Zachary},
  title = {{LogicInference}: A New Dataset for Teaching Logical Inference to seq2seq Models},
  booktitle={Proceedings of ICLR 2022 workshop on Objects, Structure and Causality},
  year={2022}
}
```

## Generating the Datasets

After checking out the code, open the `generate_dataset.py` file, and update the
`TARGET_FOLDER` variable to point to where do you want the dataset to be
generated. Then run:

`python3 generate_dataset.py`

Dataset generation might take a while, as all three splits (IID/OOD/length) are
generated in one go.

You can edit the rest of generation parameters defined under `Generation
parameters` in `generate_dataset.py` to generate different variations of the
dataset. Leave them as is, for generating the default dataset.

The dataset is generated as a set of TFRecord files in the format expected by
the T5 codebase. Basically, each example has two string features, called
`inputs` and `targets` containing the input and expected output of each example,
respectively. To generate the data in a different format, e.g. as a jsonl file,
or as a plan text file, the only function that would need to be edited is the
`generate_t5_split` function in the `generate_dataset.py` file.

Run `generate_sample_data.py` to just generate and print out some sample
examples instead.

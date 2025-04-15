# Quantifying Memorization Across Neural Language Models

Large language models (LMs) have been shown to memorize parts of their training data, and when prompted appropriately, they will emit the memorized training data verbatim.
This is undesirable because memorization violates privacy (exposing user data), degrades utility (repeated easy-to-memorize text is often low quality), and hurts fairness (some texts are memorized over others).

In our [paper](https://arxiv.org/abs/2202.07646), we describe three log-linear relationships that quantify the degree to which LMs emit memorized training data.
Memorization significantly grows as we increase (1) the capacity of a model, (2) the number of times an example has been duplicated, and (3) the number of tokens of context used to prompt the model. Surprisingly, we find the situation becomes complicated when generalizing these results across model families.
On the whole, we find that memorization in LMs is more prevalent than previously believed and will likely get worse as models continues to scale, at least without active mitigations.

This respository contains links to the prefixes and model continuations which
were used in our analysis.

## Intended Use
The prompt and generations released here are intended to be used by researchers
to study memorization in large language models as well as to develop better
mitigations for memorization.
All prompts come from [The Pile](https://arxiv.org/abs/2101.00027) and have not
not been curated for content.
Thus, the dataset should not be used outside of its intended purpose of studying
memorization.

## Prompts and Generations from GPT-Neo

### Prompt sequences

You can reproduce the prompt sequences we used in our experiments by running the following steps:

1. Download and uncompress the Pile [train set](https://mystic.the-eye.eu/public/AI/pile/train/).
2. Run the command `python3 build_pile_dataset.py PILE_DIR OUTPUT_DIR`, where `PILE_DIR` is the location of the uncompressed Pile train set, and `OUTPUT_DIR` is where you would like to write out the sequences.

The result of this script is a `prompts_{n}.npy` file for each sequence length
`n`.
Each contains an ndarray `data` where `data[b, i]` corresponds to
the `i`th sequence in the `b`th frequency bucket.

The script also outputs a `counts_{n}.npy` files for each sequence length.
Each contains an ndarray `frequencies` where
`frequencies[b, i]`
gives the exact number of times each this sequence occured in the Pile training set.
For a few rows, parsing of the original Pile data may have failed.
If `counts[b, i] == 0`, the corresponding example should be ommitted from
analysis.

**Update (9/20): Our collaborator has run this step and made the
prompts available for download
[here](https://github.com/ethz-privsec/lm_memorization_data).**

### Generations
Each row below corresponds to 10,000 generations produced using prompts
extracted from sequences of the specified sequence length using the
specified GPT-Neo model size.
For each of these configurations, we generated from several possible prefix
lengths (`p`)  between 50 tokens and `sequence length - 50` tokens long.
All generations were produced by repeatedly choosing the most
likely token predicted by the model until the generation was 50 tokens long.

Each .npy file contains an ndarray with shape
`[number of length buckets x 1000 x 50]`, where 50 is the number of tokens that
were generated.

#### 125M GPT-Neo
| Model  | Sequence Length | p=50 | p=100 | p=150 | p=200 | p=250 | p=300 | p=350 | p=400 | p=450 |
| ------ | --------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 125M   |       100       | [link](https://storage.cloud.google.com/mem-data/gens_125M/125M-0.0_prompt_50_of_100.npy) | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 125M   |       150       | [link](https://storage.cloud.google.com/mem-data/gens_125M/125M-0.0_prompt_50_of_150.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_125M/125M-0.0_prompt_100_of_150.npy) | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 125M   |       200       | [link](https://storage.cloud.google.com/mem-data/gens_125M/125M-0.0_prompt_50_of_200.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_125M/125M-0.0_prompt_100_of_200.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_125M/125M-0.0_prompt_150_of_200.npy) | n/a | n/a | n/a | n/a | n/a | n/a |
| 125M   |       250       | [link](https://storage.cloud.google.com/mem-data/gens_125M/125M-0.0_prompt_50_of_250.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_125M/125M-0.0_prompt_100_of_250.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_125M/125M-0.0_prompt_200_of_250.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_125M/125M-0.0_prompt_200_of_250.npy) | n/a | n/a | n/a | n/a | n/a |
| 125M   |       300       | [link](https://storage.cloud.google.com/mem-data/gens_125M/125M-0.0_prompt_50_of_300.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_125M/125M-0.0_prompt_100_of_300.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_125M/125M-0.0_prompt_150_of_300.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_125M/125M-0.0_prompt_200_of_300.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_125M/125M-0.0_prompt_250_of_300.npy) | n/a | n/a | n/a | n/a |
| 125M   |       350       | [link](https://storage.cloud.google.com/mem-data/gens_125M/125M-0.0_prompt_50_of_350.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_125M/125M-0.0_prompt_100_of_350.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_125M/125M-0.0_prompt_150_of_350.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_125M/125M-0.0_prompt_200_of_350.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_125M/125M-0.0_prompt_250_of_350.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_125M/125M-0.0_prompt_300_of_350.npy) | n/a | n/a | n/a |
| 125M   |       400       | [link](https://storage.cloud.google.com/mem-data/gens_125M/125M-0.0_prompt_50_of_400.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_125M/125M-0.0_prompt_100_of_400.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_125M/125M-0.0_prompt_150_of_400.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_125M/125M-0.0_prompt_200_of_400.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_125M/125M-0.0_prompt_250_of_400.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_125M/125M-0.0_prompt_300_of_400.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_125M/125M-0.0_prompt_350_of_400.npy) | n/a | n/a |
| 125M   |       450       | [link](https://storage.cloud.google.com/mem-data/gens_125M/125M-0.0_prompt_50_of_450.npy) | [link](https://storage.cloud.google.com/mem-data/gens_125M/125M-0.0_prompt_100_of_450.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_125M/125M-0.0_prompt_150_of_450.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_125M/125M-0.0_prompt_200_of_450.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_125M/125M-0.0_prompt_250_of_450.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_125M/125M-0.0_prompt_300_of_450.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_125M/125M-0.0_prompt_350_of_450.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_125M/125M-0.0_prompt_400_of_450.npy)  | n/a |
| 125M   |       500       | [link](https://storage.cloud.google.com/mem-data/gens_125M/125M-0.0_prompt_50_of_500.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_125M/125M-0.0_prompt_100_of_500.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_125M/125M-0.0_prompt_150_of_500.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_125M/125M-0.0_prompt_200_of_500.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_125M/125M-0.0_prompt_250_of_500.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_125M/125M-0.0_prompt_300_of_500.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_125M/125M-0.0_prompt_350_of_500.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_125M/125M-0.0_prompt_400_of_500.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_125M/125M-0.0_prompt_450_of_500.npy)  |

#### 1.3B GPT-Neo
| Model  | Sequence Length | p=50 | p=100 | p=150 | p=200 | p=250 | p=300 | p=350 | p=400 | p=450 |
| ------ | --------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.3B   |       100       | [link](https://storage.cloud.google.com/mem-data/gens_1.3B/1.3B-0.0_prompt_50_of_100.npy) | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 1.3B   |       150       | [link](https://storage.cloud.google.com/mem-data/gens_1.3B/1.3B-0.0_prompt_50_of_150.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_1.3B/1.3B-0.0_prompt_100_of_150.npy) | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 1.3B   |       200       | [link](https://storage.cloud.google.com/mem-data/gens_1.3B/1.3B-0.0_prompt_50_of_200.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_1.3B/1.3B-0.0_prompt_100_of_200.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_1.3B/1.3B-0.0_prompt_150_of_200.npy) | n/a | n/a | n/a | n/a | n/a | n/a |
| 1.3B   |       250       | [link](https://storage.cloud.google.com/mem-data/gens_1.3B/1.3B-0.0_prompt_50_of_250.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_1.3B/1.3B-0.0_prompt_100_of_250.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_1.3B/1.3B-0.0_prompt_150_of_250.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_1.3B/1.3B-0.0_prompt_200_of_250.npy) | n/a | n/a | n/a | n/a | n/a |
| 1.3B   |       300       | [link](https://storage.cloud.google.com/mem-data/gens_1.3B/1.3B-0.0_prompt_50_of_300.npy) | [link](https://storage.cloud.google.com/mem-data/gens_1.3B/1.3B-0.0_prompt_100_of_300.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_1.3B/1.3B-0.0_prompt_150_of_300.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_1.3B/1.3B-0.0_prompt_200_of_300.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_1.3B/1.3B-0.0_prompt_250_of_300.npy)  | n/a | n/a | n/a | n/a | n/a |
| 1.3B   |       350       | [link](https://storage.cloud.google.com/mem-data/gens_1.3B/1.3B-0.0_prompt_50_of_350.npy) | [link](https://storage.cloud.google.com/mem-data/gens_1.3B/1.3B-0.0_prompt_100_of_350.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_1.3B/1.3B-0.0_prompt_150_of_350.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_1.3B/1.3B-0.0_prompt_200_of_350.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_1.3B/1.3B-0.0_prompt_250_of_350.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_1.3B/1.3B-0.0_prompt_300_of_350.npy)  | n/a | n/a | n/a |
| 1.3B   |       400       | [link](https://storage.cloud.google.com/mem-data/gens_1.3B/1.3B-0.0_prompt_50_of_400.npy) | [link](https://storage.cloud.google.com/mem-data/gens_1.3B/1.3B-0.0_prompt_100_of_400.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_1.3B/1.3B-0.0_prompt_150_of_400.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_1.3B/1.3B-0.0_prompt_200_of_400.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_1.3B/1.3B-0.0_prompt_250_of_400.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_1.3B/1.3B-0.0_prompt_300_of_400.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_1.3B/1.3B-0.0_prompt_350_of_400.npy) | n/a | n/a |
| 1.3B   |       450       | [link](https://storage.cloud.google.com/mem-data/gens_1.3B/1.3B-0.0_prompt_50_of_450.npy) | [link](https://storage.cloud.google.com/mem-data/gens_1.3B/1.3B-0.0_prompt_100_of_450.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_1.3B/1.3B-0.0_prompt_150_of_450.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_1.3B/1.3B-0.0_prompt_200_of_450.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_1.3B/1.3B-0.0_prompt_250_of_450.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_1.3B/1.3B-0.0_prompt_300_of_450.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_1.3B/1.3B-0.0_prompt_350_of_450.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_1.3B/1.3B-0.0_prompt_400_of_450.npy) | n/a |
| 1.3B   |       500       | [link](https://storage.cloud.google.com/mem-data/gens_1.3B/1.3B-0.0_prompt_50_of_500.npy) | [link](https://storage.cloud.google.com/mem-data/gens_1.3B/1.3B-0.0_prompt_100_of_500.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_1.3B/1.3B-0.0_prompt_150_of_500.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_1.3B/1.3B-0.0_prompt_200_of_500.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_1.3B/1.3B-0.0_prompt_250_of_500.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_1.3B/1.3B-0.0_prompt_300_of_500.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_1.3B/1.3B-0.0_prompt_350_of_500.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_1.3B/1.3B-0.0_prompt_400_of_500.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_1.3B/1.3B-0.0_prompt_450_of_500.npy)  |
#### 2.7B GPT-Neo
| Model  | Sequence Length | p=50 | p=100 | p=150 | p=200 | p=250 | p=300 | p=350 | p=400 | p=450 |
| ------ | --------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2.7B   |       100       | [link](https://storage.cloud.google.com/mem-data/gens_2.7B/2.7B-0.0_prompt_50_of_100.npy) | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 2.7B   |       150       | [link](https://storage.cloud.google.com/mem-data/gens_2.7B/2.7B-0.0_prompt_50_of_150.npy) | [link](https://storage.cloud.google.com/mem-data/gens_2.7B/2.7B-0.0_prompt_100_of_150.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_2.7B/2.7B-0.0_prompt_100_of_150.npy) | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 2.7B   |       200       | [link](https://storage.cloud.google.com/mem-data/gens_2.7B/2.7B-0.0_prompt_50_of_200.npy) | [link](https://storage.cloud.google.com/mem-data/gens_2.7B/2.7B-0.0_prompt_100_of_200.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_2.7B/2.7B-0.0_prompt_150_of_200.npy)  | n/a | n/a | n/a | n/a | n/a | n/a |
| 2.7B   |       250       | [link](https://storage.cloud.google.com/mem-data/gens_2.7B/2.7B-0.0_prompt_50_of_250.npy) | [link](https://storage.cloud.google.com/mem-data/gens_2.7B/2.7B-0.0_prompt_100_of_250.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_2.7B/2.7B-0.0_prompt_150_of_250.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_2.7B/2.7B-0.0_prompt_200_of_250.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_2.7B/2.7B-0.0_prompt_200_of_250.npy) | n/a | n/a | n/a | n/a | n/a |
| 2.7B   |       300       | [link](https://storage.cloud.google.com/mem-data/gens_2.7B/2.7B-0.0_prompt_50_of_300.npy) | [link](https://storage.cloud.google.com/mem-data/gens_2.7B/2.7B-0.0_prompt_100_of_300.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_2.7B/2.7B-0.0_prompt_150_of_300.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_2.7B/2.7B-0.0_prompt_200_of_300.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_2.7B/2.7B-0.0_prompt_250_of_300.npy) | n/a | n/a | n/a | n/a |
| 2.7B   |       350       | [link](https://storage.cloud.google.com/mem-data/gens_2.7B/2.7B-0.0_prompt_50_of_350.npy) | [link](https://storage.cloud.google.com/mem-data/gens_2.7B/2.7B-0.0_prompt_100_of_350.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_2.7B/2.7B-0.0_prompt_150_of_350.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_2.7B/2.7B-0.0_prompt_200_of_350.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_2.7B/2.7B-0.0_prompt_250_of_350.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_2.7B/2.7B-0.0_prompt_300_of_350.npy) | n/a | n/a | n/a |
| 2.7B   |       400       | [link](https://storage.cloud.google.com/mem-data/gens_2.7B/2.7B-0.0_prompt_50_of_400.npy) | [link](https://storage.cloud.google.com/mem-data/gens_2.7B/2.7B-0.0_prompt_100_of_400.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_2.7B/2.7B-0.0_prompt_150_of_400.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_2.7B/2.7B-0.0_prompt_200_of_400.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_2.7B/2.7B-0.0_prompt_250_of_400.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_2.7B/2.7B-0.0_prompt_300_of_400.npy) | [link](https://storage.cloud.google.com/mem-data/gens_2.7B/2.7B-0.0_prompt_350_of_400.npy) | n/a | n/a |
| 2.7B   |       450       | [link](https://storage.cloud.google.com/mem-data/gens_2.7B/2.7B-0.0_prompt_50_of_450.npy) | [link](https://storage.cloud.google.com/mem-data/gens_2.7B/2.7B-0.0_prompt_100_of_450.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_2.7B/2.7B-0.0_prompt_150_of_450.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_2.7B/2.7B-0.0_prompt_200_of_450.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_2.7B/2.7B-0.0_prompt_250_of_450.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_2.7B/2.7B-0.0_prompt_300_of_450.npy) | [link](https://storage.cloud.google.com/mem-data/gens_2.7B/2.7B-0.0_prompt_350_of_450.npy) | [link](https://storage.cloud.google.com/mem-data/gens_2.7B/2.7B-0.0_prompt_400_of_450.npy) | n/a |
| 2.7B   |       500       | [link](https://storage.cloud.google.com/mem-data/gens_2.7B/2.7B-0.0_prompt_50_of_500.npy) | [link](https://storage.cloud.google.com/mem-data/gens_2.7B/2.7B-0.0_prompt_100_of_500.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_2.7B/2.7B-0.0_prompt_150_of_500.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_2.7B/2.7B-0.0_prompt_200_of_500.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_2.7B/2.7B-0.0_prompt_250_of_500.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_2.7B/2.7B-0.0_prompt_300_of_500.npy) | [link](https://storage.cloud.google.com/mem-data/gens_2.7B/2.7B-0.0_prompt_350_of_500.npy) | [link](https://storage.cloud.google.com/mem-data/gens_2.7B/2.7B-0.0_prompt_400_of_500.npy) | [link](https://storage.cloud.google.com/mem-data/gens_2.7B/2.7B-0.0_prompt_450_of_500.npy)  |

#### 6B GPT-Neo
| Model  | Sequence Length | p=50 | p=100 | p=150 | p=200 | p=250 | p=300 | p=350 | p=400 | p=450 |
| ------ | --------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6B   |       100       | [link](https://storage.cloud.google.com/mem-data/gens_6B/6B-0.0_prompt_50_of_100.npy) | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 6B   |       150       | [link](https://storage.cloud.google.com/mem-data/gens_6B/6B-0.0_prompt_50_of_150.npy) | [link](https://storage.cloud.google.com/mem-data/gens_6B/6B-0.0_prompt_100_of_150.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_6B/6B-0.0_prompt_100_of_150.npy) | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 6B   |       200       | [link](https://storage.cloud.google.com/mem-data/gens_6B/6B-0.0_prompt_50_of_200.npy) | [link](https://storage.cloud.google.com/mem-data/gens_6B/6B-0.0_prompt_100_of_200.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_6B/6B-0.0_prompt_150_of_200.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_6B/6B-0.0_prompt_150_of_200.npy) | n/a | n/a | n/a | n/a | n/a | n/a |
| 6B   |       250       | [link](https://storage.cloud.google.com/mem-data/gens_6B/6B-0.0_prompt_50_of_250.npy) | [link](https://storage.cloud.google.com/mem-data/gens_6B/6B-0.0_prompt_100_of_250.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_6B/6B-0.0_prompt_150_of_250.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_6B/6B-0.0_prompt_200_of_250.npy)  | n/a | n/a | n/a | n/a | n/a |
| 6B   |       300       | [link](https://storage.cloud.google.com/mem-data/gens_6B/6B-0.0_prompt_50_of_300.npy) | [link](https://storage.cloud.google.com/mem-data/gens_6B/6B-0.0_prompt_100_of_300.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_6B/6B-0.0_prompt_150_of_300.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_6B/6B-0.0_prompt_200_of_300.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_6B/6B-0.0_prompt_250_of_300.npy) | n/a | n/a | n/a | n/a |
| 6B   |       350       | [link](https://storage.cloud.google.com/mem-data/gens_6B/6B-0.0_prompt_50_of_350.npy) | [link](https://storage.cloud.google.com/mem-data/gens_6B/6B-0.0_prompt_100_of_350.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_6B/6B-0.0_prompt_150_of_350.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_6B/6B-0.0_prompt_200_of_350.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_6B/6B-0.0_prompt_250_of_350.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_6B/6B-0.0_prompt_300_of_350.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_6B/6B-0.0_prompt_300_of_350.npy) | n/a | n/a | n/a |
| 6B   |       400       | [link](https://storage.cloud.google.com/mem-data/gens_6B/6B-0.0_prompt_50_of_400.npy) | [link](https://storage.cloud.google.com/mem-data/gens_6B/6B-0.0_prompt_100_of_400.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_6B/6B-0.0_prompt_150_of_400.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_6B/6B-0.0_prompt_200_of_400.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_6B/6B-0.0_prompt_250_of_400.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_6B/6B-0.0_prompt_300_of_400.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_6B/6B-0.0_prompt_350_of_400.npy) | n/a | n/a |
| 6B   |       450       | [link](https://storage.cloud.google.com/mem-data/gens_6B/6B-0.0_prompt_50_of_450.npy) | [link](https://storage.cloud.google.com/mem-data/gens_6B/6B-0.0_prompt_100_of_450.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_6B/6B-0.0_prompt_150_of_450.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_6B/6B-0.0_prompt_200_of_450.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_6B/6B-0.0_prompt_250_of_450.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_6B/6B-0.0_prompt_300_of_450.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_6B/6B-0.0_prompt_350_of_450.npy) | [link](https://storage.cloud.google.com/mem-data/gens_6B/6B-0.0_prompt_400_of_450.npy) | n/a |
| 6B   |       500       | [link](https://storage.cloud.google.com/mem-data/gens_6B/6B-0.0_prompt_50_of_500.npy) | [link](https://storage.cloud.google.com/mem-data/gens_6B/6B-0.0_prompt_100_of_500.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_6B/6B-0.0_prompt_150_of_500.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_6B/6B-0.0_prompt_200_of_500.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_6B/6B-0.0_prompt_250_of_500.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_6B/6B-0.0_prompt_300_of_500.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_6B/6B-0.0_prompt_350_of_500.npy) | [link](https://storage.cloud.google.com/mem-data/gens_6B/6B-0.0_prompt_400_of_500.npy)  | [link](https://storage.cloud.google.com/mem-data/gens_6B/6B-0.0_prompt_450_of_500.npy)  |

### Tokenization
All of the above prompts and generations are tokenized using GPT-Neo's vocabulary.
An encoded sequence can be detokenized using the following
[HuggingFace API](https://github.com/huggingface/transformers) call:

```
from transformers import GPT2Tokenizer
vocab = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo)
print(vocab.decode(encoded_sequence))
```

## Citation
```
@inproceedings{carlini2022quantifying,
  title={Quantifying Memorization Across Neural Language Models},
  author={Carlini, Nicholas and Ippolito, Daphne and Jagielski, Matthew and Lee, Katherine and Tramer, Florian and Zhang, Chiyuan},
  booktitle={The Eleventh International Conference on Learning Representations},
  year={2022}
}
```

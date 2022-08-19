# Mostly Basic Python Problems Dataset

The benchmark consists of around 1,000 crowd-sourced Python programming problems, designed to be solvable by entry level programmers, covering programming fundamentals, standard library functionality, and so on. Each problem consists of a task description, code solution and 3 automated test cases.

As described in the paper, a subset of the data has been hand-verified by us. This data is `sanitized-mbpp.json`.

The dataset is in a .jsonl format (json per line).

Released as part of Program Synthesis with Large Language Models, Austin et. al., 2021.

## Evaluation Details

We specify a train and test split to use for evaluation. Specifically:

* Task IDs 11-510 are used for testing.
* Task IDs 1-10 were used for few-shot prompting and not for training.
* Task IDs 511-600 were used for validation during fine-tuning.
* Task IDs 601-974 are used for training.

In the paper "Program Synthesis with Large Language Models", Austin et al. 2021,
we used three-shot prompts with task_ids 2, 3, and 4 for few-shot prompts. Our prompts had the format

`You are an expert Python programmer, and here is your task: {prompt} Your code should pass these tests:\n\n{tests}\n[BEGIN]\n{code}\n[DONE]`

where the [BEGIN] and [DONE] tokens were used to delimit the model solution.

For the edited subset, the test/train/validation/prompting subsets were inherited from the above groupings.

# IFEval: Instruction Following Eval

This is not an officially supported Google product.

This repository contains source code and data for
[Instruction Following Evaluation for Large Language Models](arxiv.org/abs/2311.07911)

## Dependencies

Please make sure that all required python packages are installed via:

```
pip3 install -r requirements.txt
```

## How to run

You need to create a jsonl file with two entries: prompt and response.
Then, call `evaluation_main` from the parent folder of
instruction_following_eval. For example:

```bash
# Content of `--input_response_data` should be like:
# {"prompt": "Write a 300+ word summary ...", "response": "PUT YOUR MODEL RESPONSE HERE"}
# {"prompt": "I am planning a trip to ...", "response": "PUT YOUR MODEL RESPONSE HERE"}
# ...
python3 -m instruction_following_eval.evaluation_main \
  --input_data=./instruction_following_eval/data/input_data.jsonl \
  --input_response_data=YOUR_MODEL_RESPONSES.jsonl \
  --output_dir=./instruction_following_eval/data/
```

Responses must be generated for the exact prompts in the selected input file.
The evaluator checks prompt coverage before scoring and reports all input task
keys that have no matching response.

## Replay the archived GPT-4 demo

From the repository root, run:

```bash
bash instruction_following_eval/run.sh
```

This uses `data/input_data_gpt4_20231107_145030.jsonl` with the archived
`data/input_response_data_gpt4_20231107_145030.jsonl`. All 541 input prompts match
the saved responses. The companion input is an unchanged copy of
[`data/input_data.jsonl` at commit b7f222394aeb06af2f8f8511b910bf8a413f9339](https://github.com/google-research/google-research/blob/b7f222394aeb06af2f8f8511b910bf8a413f9339/instruction_following_eval/data/input_data.jsonl).
Its SHA-256 is
`86ea91641ee9e6b891330f3f019e82c481abddbdea5945106df1dfe59757fbc2`.

The historical input retains a known inconsistency at key `2785`: its prompt
asks for one placeholder while its `kwargs` require three. This replay restores
the input/response pairing; it does not establish reproduction of the paper's
reported scores. For new evaluations, use `data/input_data.jsonl` with responses
generated for its corrected prompts. The archived response must retain the
prompt for which it was generated.

## Reference

If you use our work, please consider citing our preprint:

```
@article{zhou2023instruction,
  title={Instruction-Following Evaluation for Large Language Models},
  author={Zhou, Jeffrey and Lu, Tianjian and Mishra, Swaroop and Brahma, Siddhartha and Basu, Sujoy and Luan, Yi and Zhou, Denny and Hou, Le},
  journal={arXiv preprint arXiv:2311.07911},
  year={2023}
}
```
# IFEval: Instruction Following Eval

This is not an officially supported Google product.

## Dependencies

Please make sure that all required python packages are installed via:

```
pip install -r requirements.txt
```

## How to run

You need to create a jsonl file with two entries: prompt and response.
Then, call `evaluation_main`. For example:

```bash

# Content of `--input_response_data` should be like:
# {"prompt": "Write a 300+ word summary ...", "response": "PUT YOUR MODEL RESPONSE HERE"}
# {"prompt": "I am planning a trip to ...", "response": "PUT YOUR MODEL RESPONSE HERE"}
# ...
python3 evaluation_main.py \
  --input_data=./data/input_data.jsonl \
  --input_response_data=./data/input_response_data_gpt4_20231107_145030.jsonl \
  --output_dir=./data/
```

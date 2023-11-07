# Instructability Eval

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

# Content of ./data/input_response_data_text_bison.jsonl:
# {"prompt": "Write a 200 word summary of ...", "response": "**\u795e\u8a71\u306e ..."}
# {"prompt": "Write an email ...", "response": "Dear [Name],\n\nI hope this email finds you well ..."}
# ...
python3 evaluation_main.py \
  --input_data=./data/input_data.jsonl \
  --input_response_data=./data/input_response_data_text_bison.jsonl \
  --output_dir=./data/
```

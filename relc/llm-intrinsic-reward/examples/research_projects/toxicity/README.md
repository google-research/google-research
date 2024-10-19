# De-detoxifying language models

To run this code, do the following:

```shell
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file {CONFIG} examples/toxicity/scripts/gpt-j-6b-toxicity.py --log_with wandb
```
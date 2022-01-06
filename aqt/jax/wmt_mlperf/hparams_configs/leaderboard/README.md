## Leaderboard Transformer Quantization

Summarizes results for leaderboard_configs.

Leaderboard configs corresponds to baseline model (bfloat16), weight quantized
models (8bit, 4bit, 2bit), and weight & activation quantized models (8bit,
4bit).

For activation quantization, we run experiments with fixed bounds as well as
dynamic bounds (automatically adjusted during training).

This directory contains a few json config files for testing ACT.

demo_config.json:
Sanity check running ACT.

sft_demo_config.json:
Sanity check running SFT.

sft_zephyr_7B_beta_config.json:
Run SFT on Zephyr 7B beta model on PACIFIC dataset with 50 conversation samples.

act_zephyr_7B_beta_very_tiny_config.json:
Sanity check running ACT on Zephyr 7B beta model that has had SFT run on it (
reads model produced by sft_zephyr_7B_beta_config.json).

act_zephyr_7B_beta_config.json:
Run ACT on Zephyr 7B beta model that has had SFT run on it (reads model produced
by sft_zephyr_7B_beta_config.json).

zephyr_7B_beta_config.json:
Run ACT on Zephyr 7B beta model without SFT (not really useful).

The data in this directory has been generated as follows:

Create post-processed train/test splits in `TFRecord` format in `/tmp` directory:

```shell
python neighborhood/data/create_neighborhood.py \
  --output_dir /tmp \
  --max_rand_len 10 --language_group felekesemitic --lang Zway \
  --random_target_algo markov --pairwise_algo lingpy \
  --task_data_dir ~/projects/ST2022/ \
  --logtostderr
```

The above generates the test split `Zway_test.tfrecords` data and the combined
symbol table `Zway.syms` in the `/tmp` directory, which are copied here.

# MAX & IMP

This is the official code release for [Alternating Gradient Descent and Mixture-of-Experts for Integrated Multimodal Perception](https://arxiv.org/abs/2305.06324), published in NeurIPS 2023.

See [this notebook](https://github.com/google-research/google-research/blob/master/imp/max/notebooks/examples.ipynb) for comprehensive examples.

## Running Instructions
To run any experiment (train/evaluation/inference) from terminal, simply run the following:

```
python3 -m max.projects.imp.main --config_name=name_of_the_config_of_interest
```

### Reproducing the Main IMP Experiments
To reproduce the main experiments (Table 1), please refer to the corresponding configuration files [here](https://github.com/google-research/google-research/blob/master/imp/max/projects/imp/config/experiment.py):

Model      |   PPT   |   IN1k   |   C100   |   K400   |   K600   |     K700   |   UCF101   |   HMDB51   |   ESC50    |          Train Config Name         |          Eval Config Name
:--------: | :-----: | :------: | :------: | :------: | :------: | :--------: | :--------: | :--------: | :--------: | :--------------------------------: | :---------------------------------:
IMP-B      |   86M   |   80.5   |   82.4   |   63.6   |   62.1   |    49.9    |    64.2    |    39.7    |    32.8    |        `imp_base.all.train`        |        `imp_base.all.eval`
IMP-MoE-B  |   90M   |   83.2   |   84.9   |   68.2   |   65.7   |    52.1    |    88.7    |    46.6    |    47.8    |  `sparse_moe_imp_base.all.train`   |   `sparse_moe_imp_base.all.eval`
IMP-MoE-L  |  350M   | **83.9** | **87.0** | **77.0** | **76.8** |  **68.3**  |  **91.5**  |  **59.1**  |  **65.1**  |  `sparse_moe_imp_large.all.train`  |   `sparse_moe_imp_large.all.eval`


### Overriding Config Parameters
In the script above, you can pass a `--config_overrides` flag that accepts any JSON-dumped string containing dictionary-style overriding instructions for the configuration settings. For example, to override the learning rate, pass the following:

```
--config_overrides='{"optimization": {"optimizer": {"learning_rate": 0.1}}}'
```
To resume training from an existing checkpoint, provide the restore path as follows:

```
--config_overrides='{"path": "path/to/experiment/dir", "optimization": {"restore_path": "path/to/restore/checkpoint"}}'
```

Refer to the `config.py` file within each directory to understand the configuration hierarchy.

### Distributed Multi-Node Training on Non-GCP Platforms

In a multi-process environment, JAX devices should be initialized with the appropriate coordinator and process configuration. This is typically handled automatically on GCP instances. However, in custom setups, the following flags should be explicitly passed:

```
--coordinator_address='IP:PORT' --process_count=NUMBER_OF_PROCESSES --process_index=HOST_SPECIFIC_PROCESS_ID
```

where:

- `--coordinator_address`: Specifies the IP address and port of the coordinator process.
- `--process_count`: Indicates the total number of processes in the multi-process setup.
- `--process_index`: Identifies the unique ID of the current process within the setup.

##### Example:

Suppose there are two GPU processes, and process 0 is the designated coordinator with address `10.0.0.1:1234`. To initialize the GPU cluster, pass the following flags.

- On process 0:

```
--coordinator_address='10.0.0.1:1234' --process_count=2 --process_index=0
```

- On process 1:

```
--coordinator_address='10.0.0.1:1234' --process_count=2 --process_index=1
```

For more details, refer to the JAX documentation on `jax.distributed.initialize` [here](https://jax.readthedocs.io/en/latest/_autosummary/jax.distributed.initialize.html).

### Distributed Data Processing

If you're using TF Data Service for data processing, pass the dispatcher's address in the following format:


```
--tf_data_service_address='[<protocol>://]<address>'
```

where:

-  `<address>` is the IP address and port (e.g., "10.0.0.1:8080")
-  `[<protocol>://]` is an optional protocol specifier (e.g., "grpc://"). If omitted, the default protocol is used.

For example, if no protocol is needed, you can simply pass:

```
--tf_data_service_address='10.0.0.1:8080'
```

For more details and examples, refer to the TensorFlow documentation on `tf.data.experimental.service.distribute` [here](https://www.tensorflow.org/api_docs/python/tf/data/experimental/service/distribute).

## References

```
@article{akbari2024alternating,
  title={Alternating gradient descent and mixture-of-experts for integrated multimodal perception},
  author={Akbari, Hassan and Kondratyuk, Dan and Cui, Yin and Hornung, Rachel and Wang, Huisheng and Adam, Hartwig},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
```


## Correspondence and Maintenance

Please feel free to report any issues to the [official repo](https://github.com/google-research/google-research) or
the corresponding author: https://github.com/hassanhub

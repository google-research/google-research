# STUDY: Socially Aware Temporally Causal Decoder Recommender Systems
Authors: Eltayeb Ahmed, Diana Mincu, Lauren Harrell, Katherine Heller,
Subhrajit Roy

This the official implementation of the paper
[STUDY: Socially Aware Temporally Causal Decoder Recommender Systems](https://arxiv.org/abs/2306.07946).

## Code base
This code base features implementations of the models Individual and STUDY from
the paper. The models are implemented in the jax ecosystem using the libraries
using the Flax stack of libraries (Jax, Flax, Optax).

## Requirements
The code was tested against python 3.10 with requirements specified
in the `requirements.txt`.
The results on the paper were produced training on TPUs using 32 TPUs per
experiments. (See paper appendix for mode details).

The code was also tested on Google Cloud with v100 GPUs using the
`gcr.io/deeplearning-platform-release/base-cu113.py310` base image.
As such the in the requirements.txt we install a version of Jax
compiled for `Cuda 11` and `cudnn8.2`.
This can be seen in line 3 of `requirements.txt`
```
jaxlib==0.4.7+cuda11.cudnn82
```
 If are trying to run this code and you have a different version of
`Cuda` or `cudnn` installed you must modify this line the requirements.txt
to match the versions of `Cuda` and `cudnn` on your system. You can find a
list of available versions of `jaxlib`.
[here](https://storage.googleapis.com/jax-releases/jax_cuda_releases.html).
Try choose the version of `jaxlib`  closest to the one in the requirements.txt
For example if you have `Cuda 12` with `cudnn 8.8` the closest version
available would be
`jaxlib==0.4.10+cuda12.cudnn88`


## Running the Code
This guide assumes the `study_recommend` folder is in your current working
directory.
All data files referred to as train data, valid data or student activity
should contain records of students reading titles and be in the format of
`data/student_activity.csv`.
All data files referred to as student info should map students to schools and
grade levels and be in the format of
`data/student_info.csv`.
### 1. Installing the requirements

Run the following command to install the requirements
```shell
pip install -r study_recommend/requirements.txt
```

### 2. Training a model
The main entry point for the code is study_recommend/study_recommend.py to
train the STUDY model on a single machine with a single GPU run the following
command.

```
python -m study_recommend.study_recommend --train_data_path=/path/to/student_activity_train.csv --valid_data_path=/path/to/student_activity_valid.csv --student_info_path=/path/to/student_info.csv --output_path=/path/to/directory/to/save/models/and/results/ --tensorboard_path=/path/to/directory/to/save/tensorboard/logs/ --student_chunk_size=65 --seq_len=65 --per_device_batch_size=2048 --n_steps=224000 --learning_rate=0.0016 --model=study
```
To train an individual model change `--model=study` to `--model=individual`

You can monitor the progress of training through the process logs
as well as through tensorboard. To view tensorboard graphs
execute the following command
```
tensorboard --logdir=/path/to/directory/to/save/tensorboard/logs/
```
and shortly after a link to a tensorboard dashboard should appear.

### 3. Choosing a batch size
Here we specify a batch size per device (per GPU) of 2048 samples (`--per_device_batch_size`). This was
chosen for TPU devices and will differ depending on the memory your devices have.
If you run on a different platform then you might be better served with a different
batch size. In practice you want the largest batch size that is a power of 2
and does not cause the code to throw an  `XlaRuntimeError: RESOURCE_EXHAUSTED`
error. If the code throws this error then you must half the batch size. If the
code starts training successfully you can try doubling the batch size. You can
know training has started successfully when you see the log message

```
'Doing reference evals .....
```
If you get an `XlaRuntimeError: RESOURCE_EXHAUSTED`
you can half the batch size. Do this till you find the largest batch size that
works.

#### Running in a parallel environment
If you have multiple GPUs to get the best training speed during training we
recommend you launch one process per GPU/TPU even if all GPUs/TPUs are on the
same  machine. If you have multiple machines and are doing distributed training
we recommend that on each machine you launch one process per GPU/TPU on the
machine. Launching multiple processes is usually handled by tools such as
SLURM `srun` and Open MPI `mpirun/mpiexec`. This is especially useful for
launching multi-machine jobs. If you are using one of these tools then you
need to add the following flag to your training job run
`in_managed_parallel_env`.
For more details on supported tooling see
[here](https://jax.readthedocs.io/en/latest/multi_process.html)

If you do not have access to such tooling then you will need to supply
the following flags
1. `--num_processes=<total number of processes>` for all processes.
2.  `--process_id=<i>` where `i` is a unique index for each process going from
`0` to `num_processes-1`
3.  `--coordinator_address=<ip:port>` where `ip` is the ip address of the process
with `process_id=0` (can `127.0.0.1` if all jobs are on the same machine) and
`port` is a free port on that machine. This flag is the same for all processes.

If all jobs are on the same machine we provide a utility script to launch
one job per GPU on the same machine. To use this script to launch a training job
do the following
```shell
python -m study_recommend.launch_parallel <training_flags> --num_processes=<number_of_GPUs> --coordinator_address=127.0.0.1:4321
```
Where <training_flags> are the flags you want to pass to
`study_recommend.study_recommend` and <number_of_GPUs> is the number of GPUs on
the machine.  The `study_recommend.launch_parallel `utility script will then
launch one process per GPU and will automatically populate the `--process_id`
flag.

#### Scaling to your new overall batch Size
Your overall batch size is equal to `per_device_batch_size` multiplied by the
total number of devices (GPUs or TPUs) across all hosts.
The command above assumes a global batch size of 2048 If your overall batch size
is different we recommend you adjust the number of steps (`--n_steps`)and learning rate
(`--learning_rate`) to in the following way.

1. Compute your scaling ratio `r` using the following formula

  ```
  new_overall_batch_size = per_device_batch_size * number_of_gpus_or_tpus
  r = new_overall_batch_size / 2048
  ```

2. Divide `n_steps` by the ratio `r`

  ```
  new_n_steps = 224000 / r
  ```

3. Multiply the learning rate by the ratio `r`

  ```
  new_learinng_rate=0.0016 * r
  ```

4. Replace the values of `--n_steps` and `--learning_rate` with the final
numbers computed in 2 and 3.

### 4. Doing inference with a trained model
1. Install the requirements if you have not done so already.

2. To generate recommendations from a trained model you will need to the
  model checkpoint folder, the model config, and the vocabulary. You will find all
  of these in the directory you supplied in as output_path to the training script.
  Your folder structure should look something like this

  ```
  output_path/
  ├── checkpoints
  │   ├── checkpoint_000000000000000
  │   ├── checkpoint_000000000000001
  │   ├── checkpoint_000000000000002
  │   ├── checkpoint_000000000000003
  │   ├── checkpoint_000000000000004
  │   ├── checkpoint_000000000000005
  │   └── checkpoint_000000000000006
  ├── aggregate.csv
  ├── config.json
  ├── per_student.csv
  ├── recommendations.json
  └── vocab.json
  ```

  First we load the model and vocabulary.

  ```python
  from study_recommend.utils import restore_utils

  config_file_path = "output_path/config.json"
  vocab_path = "output_path/vocab.json"
  # Load model
  params, model_eval_config, model_class, experiment_cfg = restore_utils.restore_model(config_file_path)
  # Load the vocabulary
  vocab = restore_utils.restore_vocabulary(vocab_path)
  ```
3. To do inference from data stored in files. Bulk Inference will be faster on
machines with a GPU/TPU.

  ```python
  from study_recommend import datasource as datasource_lib
  from study_recommend import inference
  from study_recommend.utils import load_data
  import pandas as pd
  student_activity_path = "data/student_activity.csv"
  student_info_path = "data/student_info.csv"
  N_RECOMMEDATIONS = 20
  PER_DEVICE_BATCH_SIZE = 2048
  eval_data = load_data.load_student_activity_files(student_activity_path)
  with open(student_info_path) as f:
    student_info = pd.read_csv(f)
  eval_datasource, _ = (
        datasource_lib.ClassroomGroupedDataSource.datasource_from_grouped_activity_dataframe(
            eval_data,
            student_info,
            seq_len=experiment_cfg.max_target_length,
            student_chunk_len=experiment_cfg.max_target_length,
            vocab=vocab,
            classroom="school_year",
            with_replacement=False,
            ordered_within_student=True,
            max_grade_level=experiment_cfg.num_grade_levels
        )
    )
  recommendations =  inference.recommend_from_datasource(
        model_eval_config,
        model_class,
        params,
        eval_datasource,
        n_recommendations=N_RECOMMEDATIONS,
        vocab=vocab,
        per_device_batch_size=PER_DEVICE_BATCH_SIZE,
    )
  ```
  Recommendations will be a dictionary mapping StudentIDs to a list of
  recommendations, one per item read. Each recommendation will be a list of
  N_RECOMMENDATION titles sorted from most highly recommended to
  least recommended.



4. To manually compose a sample and feed it through the model

  ```python
  import jax
  import jax.numpy as jnp
  import numpy as np
  from study_recommend import types


  assert vocab['<SEP>'] != vocab.oov_value
  separator_token = vocab['<SEP>']
  # The titles read by two students in the samle classroom.
  student_1_titles = ['title_1', 'title_2', 'title_3']
  student_2_titles = ['title4', 'title_5']

  student_1_titles = list(map(vocab.encode, student_1_titles))
  student_2_titles = list(map(vocab.encode, student_2_titles))

  titles = [student_1_titles + [separator_token] + student_2_titles]

  # The student id for each interaction above.
  student_1_student_id = [1, 1, 1]
  student_2_student_id = [2, 2]
  student_ids = [student_1_student_id + [separator_token] + student_2_student_id]

  # Integers representing the timestamps of the above interactions.
  student_1_timestamps = [100, 200, 300]
  student_2_timestamps = [200, 300]
  timestamps = [student_1_timestamps + [separator_token] + student_2_timestamps]

  # Order the above titles were read in (relative to each student).
  # These are always lists that go 0 -> n.
  student_1_input_positions = [0, 1, 2]
  student_2_input_positions = [0, 1]
  # Input positions are seperated by zeros.
  input_positions = [student_1_input_positions + [0] + student_2_input_positions]
  # The joint grade level for both students.
  grade_level = [4]

  # Cast model inputs to jax Arrays.
  titles = jnp.array(titles, dtype=jnp.int32)
  student_ids = jnp.array(student_ids, dtype=jnp.int32)
  # This may cause issues in 2038 when the timestamps overflow int32.
  timestamps = jnp.array(timestamps, dtype=jnp.int32)
  input_positions = jnp.array(input_positions, dtype=jnp.int32)
  grade_level = jnp.array(grade_level, dtype=jnp.int32)

  fields = types.ModelInputFields
  # Package up inputs into a dictionary.
  model_input = {
    fields.TITLES: titles,
    fields.INPUT_POSITIONS: input_positions,
    fields.STUDENT_IDS: student_ids,
    fields.TIMESTAMPS: timestamps,
    fields.GRADE_LEVELS: grade_level
  }
  # Run data through model.
  model = model_class(model_eval_config)
  logits =  model.apply({'params': params}, model_input)
  # Get top 20 recommendations from logits.
  _, top_20_recommendations = jax.lax.top_k(logits, 20)
  top_20_recommendations = np.asarray(top_20_recommendations)
  top_20_recommendations = np.vectorize(vocab.decode)(top_20_recommendations)
  ```

### Note on inference.
When doing inference either from arrays or from files, we will produce
`t` recommendations for each student, where `t` is the number of titles the
student has interacted with.  These are recommendations for the`t` points in the
past as can be scored for accuracy. When deploying we are interested in making
a recommendation for the point in time `t + 1`. The easiest way to do this
would be to add a fake record for the student we want to make a recommendation
for, with a timestamp far in the future. The title we use for this fake
record does not make a difference as the model will not look at it.
These can be added to the `student_activity.csv` file as fake records if using
step 3 or the student arrays if using step 4.

#### Experimental Grade Level Conditioning
To make the models condition on grade levels add the following flag to your train
job `--experimental_num_grade_levels=i` where `i` is the highest grade level.
This makes the models less likely to recommend out of grade level material
to new students or new classrooms. Our initial experiments found it lead
to a moderate increase in overall scores of `Individual` and lead to minor
decrease in overall scores of `Study`. This is not surprising as STUDY can
already use reading activity of classmates to infer the grade level.

### Notes
1. This code comes with unit tests to aid development. To run the unit tests run the following command:
```
python -m study_recommend.unittest_runner
```

2. We use signed int32 to represent all integers as not all jax installations
support int64. This is potentially problematic since we represent timestamps as
the number of whole seconds since the epoch. This means the `StudyRecommender`
might not function correctly if the activity data contains timestamps with dates
beyond 2037. We made this choice to minimise compatibility issues.
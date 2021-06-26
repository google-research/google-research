This directory contains the TF code for training and evaluating the neural
attention-based encoder-decoder model for estimating the logography of a writing
system.

## Prerequisites

You will need `python3` and `virtualenv`. Install the necessary requirements in
a virtual environment:

```shell
> virtualenv -p python3 .
> source ./bin/activate
> pip3 install -r homophonous_logography/requirements.txt
```

## Examples

In what follows, we assume that the preprocessed Bible datasets are stored in
directory `$BIBLES_DIR`. The datasets are cloned from their github
[repo](https://github.com/christos-c/bible-corpus) and preprocessed as described
in the paper.

1.  Training only:

    ```shell
    > python3 -m homophonous_logography.neural.train_or_eval_main \
      --datasets_dir ${BIBLES_DIR} \
      --model_type RNN \
      --noeval --languages finnish --model_dir /tmp/logo-models --logtostderr
    ```

1.  Evaluation only:

    ```shell
    > python3 -m homophonous_logography.neural.train_or_eval_main \
      --datasets_dir ${BIBLES_DIR} \
      --model_type RNN \
      --eval --notrain --languages finnish --model_dir /tmp/logo-models \
      --logtostderr
    ```

1.  Both:

    ```shell
    > python3 -m homophonous_logography.neural.train_or_eval_main \
      --datasets_dir ${BIBLES_DIR} \
      --eval --languages finnish --model_dir /tmp/logo-models --logtostderr
    ```

1.  Train and test multiple languages:

    ```shell
    > python3 -m homophonous_logography.neural.train_or_eval_main \
      --datasets_dir ${BIBLES_DIR} \
      --model_type RNN \
      --eval --train \
      --languages english,chinese,finnish --model_dir /tmp/logo-models \
      --logtostderr
    ```

# About
This is code for "Gradient-Based Language Model Red Teaming," N. Wichers et al. (EACL 2024) accessible [here](https://arxiv.org/abs/2401.16656)

## Implementation

Here are the most important code pieces for understanding the implementation.

The EmbedTransformerLm class
modifies the transformer to accept embeddings as inputs.

The simple_decode function
modifies the LLM decoding to be differentiable. It uses the Gumbel
softmax to sample in between decoding steps.

The loss_fn function
combines all of the pieces together. It feeds the prompt probabilities into the
LM, then runs decoding. Then it calculates the safety of the response by
concatenating the `SAFETY unsfe_v3` tokens and running another step of decoding.
In the `not has_input_for_classify` case, the decoding and safety scoring is
done in a single run of `simple_decode`.

# Setup
Use python3.10.
Modify requirements.txt to install the correct version of JAX and TF depending on your hardware. See https://github.com/jax-ml/jax#installation for details.
Install from requirements.txt as described in https://pip.pypa.io/en/stable/cli/pip_freeze/#examples

At this point you should be able to run the code by following the steps in the Running section, but the output will be meaningless since the model is randomly initialized.

Train a PAX LM by following the instructions [here](https://github.com/google/paxml/tree/main#example-convergence-runs). The model should be trained to predict a response after the 'SEP' token, and predict the safety score after the 'SAFETY' token.
Change the exp.checkpoint_dir in the config to the checkpoint directory of this model. Also change the self._vocabulary in experirment.py to the vocab your model uses.

# Running
To train locally: `python3.10 experiment.py --config=configs/model_diff_loss_config.py`

You can view the training curves in tensorboard.

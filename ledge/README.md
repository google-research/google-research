# Ledge

This is the code associated with our paper "Interpretable Actions:
Controlling Experts with Understandable Commands" by Shumeet Baluja,
David Marwood, and Michele Covell.  The paper is published in the
thirty-fifth Conference on Artificial Intelligence (AAAI-21).

This code implements the Function Approximator example from the paper,
approximately reconstructing a 1-D function. Following the paper, the
results of step 1 and step 2 are provided in the ckpts directory. This
code performs:

  Step 3: Correct the residual error.
  Step 4: Extract the interpretable commands. Commands are printed
    to the console.
  Step 5: Fine-tune the extracted commands.

To install, untar the data package in the same directory as ledge.py:

```
pip3 install -r requirements.txt
tar zxf data.tar.gz
```

As a simple demo:
```
python -m ledge --target_function_file targets/approximate_20
eog /tmp/refit_fig0.png
```

For a list of flags and options:
```
python -m ledge --help
```

# Group Testing: doing more with less.

Implementation of (the group testing work)[https://arxiv.org/abs/2004.12508]
using Sequential Monte-Carlo techniques.

We offer two modes to run our code.

The first one is an interactive mode designed to be used by a wet lab. The user configures an initial setup using a gin config file (see configs/toy.gin for example). This files defines the number of patients, eventual priors on their infection, and prior parameters of the PCR machine such as the specificity and sensitivity (can be group size dependent), the size of the groups or the number of stages. In this mode, the user will then be sequentially offered on the command line, groups of patients to be simultaneously tested and will be ask to enter the ones that were tested as positive by the PCR machine. Those tests results will be used by our algorithm to compute and return new optimal groups to be tested in the new batch. At each iteration, we also return the marginal of each individual, that is to say our current belief of the diseased status of each patient, based on the initial prior and the different test results.

To run this first mode:
```
python3 -m run_experiment --gin_config=configs/toy.gin --interactive_mode
```

The second mode is designed to assess the performance of a given group policy, either one of policies implemented in our package or a new one. The second mode simulates wet lab results, with a configurable sensitivity and specificity (can be per group size) and a prior infection rate (can be per individual) and keeps running simulations in order to build precision-recall or ROC curves, or compute the expected sensitivity / specificity of a given group testing strategy in a particular setup. This mode is more expensive to run but is very useful in order to compare the performance of different methods in different conditions: different infection rate regimes or different level of noises in PCR results for instance.

To run this second mode and assess the performance of a group testing strategy:
```
python3 -m run_experiment --gin_config=configs/toy.gin --nointeractive_mode
```

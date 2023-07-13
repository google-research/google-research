# Robots That Ask For Help: Uncertainty Alignment for Large Language Model Planners

Project website: https://robot-help.github.io/

**Abstract** Large language models (LLMs) exhibit a wide range of promising capabilities --- from step-by-step planning to commonsense reasoning --- that may provide utility for robots, but remain prone to confidently hallucinated predictions. In this work, we present KnowNo, a framework for measuring and aligning the uncertainty of LLM-based planners, such that they know when they don't know, and ask for help when needed. KnowNo builds on the theory of conformal prediction to provide statistical guarantees on task completion while minimizing human help in complex multi-step planning settings. Experiments across a variety of simulated and real robot setups that involve tasks with different modes of ambiguity (for example, from spatial to numeric uncertainties, from human preferences to Winograd schemas) show that KnowNo performs favorably over modern baselines (which may involve ensembles or extensive prompt tuning) in terms of improving efficiency and autonomy, while providing formal assurances. KnowNo can be used with LLMs out-of-the-box without model-finetuning, and suggests a promising lightweight approach to modeling uncertainty that can complement and scale with the growing capabilities of foundation models.

## Instructions

We provide three self-contained Colab:

* KnowNo-Demo - Quick demo showing generating the prediction set in the mobile manipulation setting (using pre-run calibration results)
* KnowNo-MobileManipulation - Full calibration and testing setup in the mobile manipulation setting
* KnowNo-TableSim - Full calibration and testing setup in the table rearrangement setting, also running the PyBullet simulation.

Minimal dependencies (openai, pybullet) are required.

## Disclaimer

The colabs use the [GPT-3.5](https://arxiv.org/abs/2005.14165) (text-davinci-003) model, which underperforms [PaLM2-L](https://ai.google/discover/palm2/) model used in our experiments, largely due to its bias towards option C and D over option A and B in multiple choice question answering. We also find such bias dependent on the context, so adjusting bias for certain options in the API call does not help significantly.
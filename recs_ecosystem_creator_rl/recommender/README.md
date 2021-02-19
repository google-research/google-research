## Recommender optimizing user and creator utilities.

This directory implements the recommender in ecosystem to optimize both user and
creator utilities. For details, please refer to the recommender design described in the paper.

Specifically,

-   value_model.py: Implement RNN value models to estimate utility of current
    policy based on history, including both UserValueModel and
    CreatorValueModel. More importantly, these value models learn hidden states
    from the history, which will be fed into agent to generate recommendations.

-   agent.py: Implement random agent and policy gradient agent. The policy
    gradient agent takes inputs of document features, user hidden states(learned
    from user_value_model), and creator hidden states(learned from
    creator_value_model).

-   runner.py: A runner class to run simulations given ecosystem environment and
    an agent.

-   data_utils.py: Helper functions to restore and format data. The
    ExperienceReplay class saves simulation data and format it into traning
    batch data to update value models and agent. Other functions include getting
    user hidden states from user history using user_value_model, getting creator
    hidden states from creator history using creator_value_model, aligning
    documents with their corresponding creators in a candidate set to featurize
    actions.

-   model_utils.py: Modules used in building NN models.

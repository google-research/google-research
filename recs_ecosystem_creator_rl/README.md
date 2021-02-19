Code for paper: "Towards Content Provider-Aware Recommendation Systems:
A Simulation Study on Interplays among User and Provider Utilities"
accepted to the Web conference 2021.

Authors:
Ruohan Zhan, Konstantina Christakopoulou, Elaine Le, Jayden Ooi,
Martin Mladenov, Alex Beutel, Craig Boutilier, Ed Chi, Minmin Chen.

For a quick start, see experiment/ecoagent_experiment.py for how to initialize a
gym ecosystem environment, a user utility model, a creator utility model,
an actor, and how to collect data using the agent and train the agent.

The code is organized into the following subdirectories:
* environment: implementation of a gym ecosystem based on RecSim.
This ecosystem consists of users, creators, and an agent, and models the
interaction among these three parties.
- To create a gym environment, call
`env = environment.create_gym_environment(env_config)`.
An instance of env_config can be found in environment.ENV_CONFIG.

- To adjust the environment setup, change the corresponding hyperparameters in
the env_config. To see the definition of different hyperparameters, please refer
to the creator.DocumentSampler() and user.UserModel().
Interesting hyperparameters may include:
-- creator_recommendation_reward
-- creator_user_click_reward
-- creator_is_saturation
-- creator_topic_influence
(whether creator.topic_preference is influenced by the user-consumed documents.)
-- copy_varied_property (whether to create two identical creator groups)

- To change the user reward function, see user.py in one of the following:
-- UserState.score_document()
-- UserModel.create_response()

* recommender: implementation of user/creator RNN utility models
(both in value_model.py) and policy gradient actor model in agent.py.

* experiment: implementation of running RandomAgent(value_model_experiment.py)
and EcoAgent(ecoagent_experiment.py).

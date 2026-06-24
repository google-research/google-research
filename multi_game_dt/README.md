# Multi-Game Decision Transformers


Work in progress:

"Multi-Game Decision Transformers"
Kuang-Huei Lee*, Ofir Nachum*, Mengjiao Yang, Lisa Lee, Daniel Freeman, Winnie Xu, Sergio Guadarrama, Ian Fischer, Eric Jang, Henryk Michalewski, Igor Mordatch*

[Paper](https://arxiv.org/abs/2205.15241) | [Blog](https://ai.googleblog.com/2022/07/training-generalist-agents-with-multi.html)

## Abstract


A longstanding goal of the field of AI is a strategy for compiling diverse experience into a highly capable, generalist agent.
In the subfields of vision and language, this was largely achieved by scaling up transformer-based models and training them on large, diverse datasets.
Motivated by this progress, we investigate whether the same strategy can be used to produce generalist reinforcement learning agents.
Specifically, we show that a single transformer-based model -- with a single set of weights -- trained purely offline can play a suite of up to 46 Atari games simultaneously at close-to-human performance.
When trained and evaluated appropriately, we find that the same trends observed in language and vision hold, including scaling of performance with model size and rapid adaptation to new games via fine-tuning.
We compare several approaches in this multi-game setting, such as online and offline RL methods and behavioral cloning, and find that our Multi-Game Decision Transformer models offer the best scalability and performance.



## Additional information

Videos and additional information can be seen at https://sites.google.com/view/multi-game-transformers


## Colab

The [colab](Multi_game_decision_transformers_public_colab.ipynb) loads a checkpoint
and evaluates on a given Atari game.

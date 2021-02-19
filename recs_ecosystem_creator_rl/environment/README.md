## Ecosystem environment.

This directory implements the ecosystem simulation environment. For details,
please refer to the simulation environment described in the paper.

Specifically,

-   user.py: Implement user model in ecosytem, including UserState(user state
    representation), UserSampler(sample a user), ResponseModel(user response
    representation), UserModel(user dynamics);
-   creator.py: Implement creator model, including Document(document
    representation), Creator(creator state representation and transition
    dynamics), DocumentSampler(sample documents from the document pool).
-   environment.py: Implement a recsim environment and a gym environment for the
    designed ecosystem.
-   sampling_utils.py: Sampling helpers to sample from a simplex, a unit ball,
    and a truncated normal distribution.

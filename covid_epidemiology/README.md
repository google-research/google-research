This repository corresponds to the paper [A prospective evaluation of AI-augmented epidemiology to forecast COVID-19 in the USA and Japan](https://www.researchsquare.com/article/rs-312419/v1).

The tf_seir.py file contains the main functions to define, train and evaluate
the proposed model, which integrates learnable encoders into compartmental
(SEIR-extended) models. We have separate models for US country, state and
country level models, as well as Japan prefecture-level model. Each of these
have specific functions and config files, with names
'generic_seir_XXX_constructor.py' and 'generic_seir_specs_XXX.py'.
The encoder-related functions are in the 'encoders' directory.

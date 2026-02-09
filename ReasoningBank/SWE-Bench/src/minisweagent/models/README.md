# LM interfaces

* `litellm_model.py` - Wrapper for [Litellm](https://github.com/BerriAI/litellm) models
   (should support most of all models).
* `anthropic.py` - Anthropic models have some special needs, so we have a separate interface for them.
* `test_models.py` - Deterministic models that can be used for internal testing
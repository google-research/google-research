# LocoProp: Enhancing BackProp via Local Loss Optimization  
**Ehsan Amid, Rohan Anil, Manfred Warmuth - AISTATS 2022**  
https://proceedings.mlr.press/v151/amid22a/amid22a.pdf


Second-order methods have shown state-of-the-art performance for optimizing deep neural networks. Nonetheless, their large memory requirement and high computational complexity, compared to first-order methods, hinder their versatility in a typical low-budget setup. LocoProp introduces a general framework of layerwise loss construction for multilayer neural networks that achieves a performance closer to second-order methods while utilizing first-order optimizers only. Our methodology lies upon a three-component loss, target, and regularizer combination, for which altering each component results in a new update rule. We provide examples using squared loss and layerwise Bregman divergences induced by the convex integral functions of various transfer functions. Our experiments on benchmark models and datasets validate the efficacy of our new approach, reducing the gap between first-order and second-order optimizers.


![LocoProp](https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEjX5vceZXAWIJohaqhy5tPqs52ryTd78pxjlGiF4qOkAdTZ2tA_2nCFX2lFYJSqAHyWvXG_3vSwix6YhQPQlHLYcEN8JxrC-P-E2nK1b5oSKCqbST5AisTpmo8p0F0xN7UaKfErkit2juHxHc7U4TCEBiNtBzORZ0fpCFv4IK7k_aVj5_1VaBQ8mOjW0w/s16000/image1.gif)  
Similar to backpropagation, LocoProp applies a forward pass to compute the activations. In the backward pass, LocoProp sets per neuron "targets" for each layer. Finally, LocoProp splits model training into independent problems across layers where several local updates can be applied to each layer's weights in parallel. See our [Google AI blog post](https://ai.googleblog.com/2022/07/enhancing-backpropagation-via-local.html) for further details.

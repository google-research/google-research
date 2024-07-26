Implementation of Polysketch attention described in
"PolySketchFormer: Fast Transformers via Sketching Polynomial Kernels" Praneeth Kacham, Vahab Mirrokni, Peilin Zhong. The paper is available at https://arxiv.org/abs/2310.01655

# Description
Transformer class in model.py can be used to create a decoder-only transformer
model. Four types of attention are currently supported:

1. **Softmax**: Vanilla attention implementation using flax.linen.dot_product_attention (quadratic time)

2. **Polynomial**: Polynomial attention with arbitrary even degree (quadratic time)

3. **Random Sketch**: Samples an independent degree 2 sketch of Ahle et al., for each transformer layer and applies tensorization trick to obtain a degree 4 sketch (linear time)

4. **Learned Sketches**: The sketches are now two level neural networks whose parameters are learned along with the rest of the parameters of the network (linear time)

**(Coming soon)** Mixed Sketch: Exact local polynomial attention and polysketch attention globally (linear time)



# Usage

## Instantiate an object of the dataclass model.TransformerConfig
Create an instance of the dataclass ```model.TransformerConfig``` setting the following attributes:

- ```vocab_size```: Vocabulary size of the transformer model
- ```context_length```: Max context length of the model. Usually powers of 2 such as 512, 1024, 2048, ...
- ```emb_dim```: Model dimension
- ```num_heads```: Number of attention heads in each layer
- ```num_layers```: Number of transformer layers
- ```dropout_rate```: Dropout rate to be used in the dense layers
- ```attention```: ```'softmax'```, ```'polynomial'```, ```'random_sketch'```, ```'learned_sketch'```
- ```power```: Degree to be used in 'polynomial' attention
- ```sketch_size```: The parameter ```r``` in the paper. Typical values are 32 and 64
- ```grain_size```: Size of blocks to be used in the lower triangular matrix multiplication algorithm. Typical sizes are 256, 512, 1024 depending on the accelerator.
- ```sketch_key```: Random key to be used to generate sketches when attention == 'random_sketch'
- ```checkpoint_attention```: Whether to recompute attention during the backward pass. Set to ```True``` to decrease the memory requirement for training.

## Create the Transformer model
Instantiate the Transformer model using ```model.Transformer(config)```, where ```config```
is the object created above.

## Train
Train the model using your favorite JAX training pipeline. An example training pipeline is available at https://github.com/google/flax/tree/main/examples/lm1b

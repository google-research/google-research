# Felix config

For information about attention_probs_dropout_prob, hidden_act, hidden_dropout_prob, hidden_size, initializer_range, intermediate_size, max_position_embeddings, num_attention_heads, num_hidden_layers, type_vocab_size, vocab_size. Please refer to [BERT](https://github.com/google-research/bert).

## num_classes
Is the number of possible tags, if non default settings are used this should be set to the number of tags found within the label_map.json file + 2.


## query_size

The dimensions of the Query/Key pair used for the pointing (re-ordering) network.

## query_transformer

If (and how many) additional transformer layers should be used for the pointing (re-ordering) network. This can be set to False or given a positive integer value.

## pointing

Currently setting this to False is not supported.

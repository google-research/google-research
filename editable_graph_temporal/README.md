# Editable Graph Temporal Model

Code for editable graph temporal model --- GATRNN, which can jointly learn to infer relational graph and forecast multivariate time series. It is designed to be easily edited from user's feedback on the predicted relational graphs.

## Data

In this code, the raw data file is a .npz file produced by the savez function of numpy. In the file, a dictionary-like object is saved, where the time series array (shape: seq_len x num_nodes x num_features) can be indexed by the key 'x' and the adjacency matrix array (shape: num_nodes x num_nodes) can be indexed by the key 'adj'. Please save your data in this format when adding new data if possible.

## Usage

Run model training:

```
python -m script_train -s /path/to/save/results
```

Run model editing:

```
python -m script_edit -mp /path/to/trained/model -s /path/to/save/results
```
